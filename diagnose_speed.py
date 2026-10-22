"""Frozen-model movement diagnostics for the original FedContextTrust_V2X v1.

Place beside pyproject.toml (project root). Uses existing prepared windows and
model.pt; never trains, overwrites weights, or changes StreamingDetector.
Run --list-prepared data first, then see --help.

Task 2: six component errors, speed-only error, benign-scaled maximum error.
Task 3: speed/position consistency using consecutive same-sender observations.
Reference scaling uses score_ref; thresholds use calibration; Test is evaluation
only. Defaults retain the saved run's q_low/q_high (normally .95/.99).
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if (ROOT / 'src' / 'fedcontext').is_dir():
    sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import pandas as pd

FEATURES = ['dx', 'dy', 'speed', 'acceleration', 'heading_sin', 'heading_cos']
SCALES = np.array([10., 10., 40., 10., 1., 1.])


def list_prepared(root):
    found = False
    for p in sorted(Path(root).rglob('preparation.json')):
        if not (p.parent / 'windows.npz').is_file():
            continue
        found = True
        info = json.loads(p.read_text(encoding='utf-8'))
        meta = pd.read_csv(p.parent / 'windows.csv', usecols=['partition', 'variant'])
        print('\nPREPARED FOLDER:', p.parent)
        print('Partitions:', meta.partition.value_counts().to_dict())
        print('TEST variants:', sorted(meta.loc[meta.partition.eq('test'), 'variant'].unique()))
        print('Origin:', info.get('origin'))
    if not found:
        raise ValueError('No prepared folders found. Supply the parent directory containing windows.npz.')


def load(path, run_info):
    from fedcontext.prepare import load_prepared
    data, meta, info = load_prepared(path)
    if info['features']['movement'] != FEATURES or info['features']['timing'][1] != 'send_gap':
        raise ValueError('This script expects the ORIGINAL six-feature movement implementation.')
    if not np.allclose(info['fixed_scales']['movement'], SCALES):
        raise ValueError('Movement scaling differs from v1; do not mix feature versions.')
    if info['road_map'] != run_info['road_map']:
        raise ValueError('Model and prepared road-map modes differ.')
    for k in ('history', 'max_gap_s', 'neighbor_ttl_s', 'neighbor_radius_m'):
        if info['config'][k] != run_info['config'][k]:
            raise ValueError(f'Model/prepared configuration mismatch: {k}')
    if data['movement'].shape[2] != 6 or data['timing'].shape[2] != 7:
        raise ValueError('Unexpected input dimensions.')
    if not meta.label.isin([0, 1]).all():
        raise ValueError('Expected binary labels 0 and 1.')
    return data, meta, info


def select(meta, name, benign=False):
    ids = np.flatnonzero(meta.partition.eq(name))
    if not len(ids):
        raise ValueError(f'No {name} windows. Select the correct prepared folder with --list-prepared.')
    if benign:
        clean = meta.iloc[ids].clean_window.astype(str).str.lower().isin(['true', '1'])
        if not meta.iloc[ids].label.eq(0).all() or not clean.all():
            raise ValueError(f'{name} contains attacked/unclean windows; refusing calibration.')
    return ids


def compute(model, data, ids, info, batch_size):
    import torch
    predicted = []
    with torch.inference_mode():
        for start in range(0, len(ids), batch_size):
            idx = ids[start:start + batch_size]
            x = torch.from_numpy(data['movement'][idx]).float()
            predicted.append(model.movement(x).cpu().numpy())
    pred = np.concatenate(predicted).astype(np.float64)
    actual = data['target'][ids].astype(np.float64)
    errors = (pred - actual) ** 2
    if not np.isfinite(errors).all():
        raise ValueError('Non-finite neural prediction errors.')
    physical_actual, physical_pred = actual * SCALES, pred * SCALES
    previous_speed = data['movement'][ids, -1, 2].astype(float) * SCALES[2]
    # target dx/dy is the current-minus-previous displacement. The final timing
    # row includes the current message, so its send_gap spans this same interval.
    dt = data['timing'][ids, -1, 1].astype(float) * info['fixed_scales']['timing'][1]
    distance = np.hypot(physical_actual[:, 0], physical_actual[:, 1])
    avg_reported = (previous_speed + physical_actual[:, 2]) / 2
    valid = np.isfinite(dt) & (dt > 1e-6) & (dt <= info['config']['max_gap_s'])
    estimated = np.full(len(ids), np.nan)
    estimated[valid] = distance[valid] / dt[valid]
    residual = np.abs(estimated - avg_reported)
    # Constant-acceleration approximation: end speed ~= interval speed + a*dt/2.
    # Unlike the midpoint check, this does not average two possibly opposite
    # random speed offsets. Acceleration remains an untrusted reported feature.
    endpoint_estimated = estimated + 0.5 * physical_actual[:, 3] * dt
    endpoint_residual = np.abs(endpoint_estimated - physical_actual[:, 2])
    # No deletion of large residuals: they may be attacks. Invalid intervals
    # stay missing and are reported as Uncertain, never as zero error.
    return dict(errors=errors, pred=physical_pred, actual=physical_actual,
                dt=dt, previous_speed=previous_speed, estimated=estimated,
                avg_reported=avg_reported, residual=residual,
                endpoint_estimated=endpoint_estimated, endpoint_residual=endpoint_residual)


def make_scores(result, feature_scales, physics_scale):
    e = result['errors']
    scaled_max = (e / feature_scales).max(axis=1)
    physics = result['residual'] / physics_scale[0]
    endpoint = result['endpoint_residual'] / physics_scale[1]
    return {
        'movement_original_mean': e.mean(axis=1),
        **{f'error_{name}': e[:, i] for i, name in enumerate(FEATURES)},
        'movement_calibrated_max': scaled_max,
        'speed_position_consistency': physics,
        'speed_position_endpoint': endpoint,
    }


def evaluate(meta, scores, thresholds, common_only=False):
    from sklearn.metrics import roc_auc_score, average_precision_score
    common = np.logical_and.reduce([np.isfinite(s) for s in scores.values()])
    groups = [('ALL', np.ones(len(meta), dtype=bool))]
    groups += [(str(v), meta.variant.eq(v).to_numpy()) for v in sorted(meta.variant.unique())]
    rows = []
    y = meta.label.to_numpy(dtype=int)
    for variant, group in groups:
        for name, s in scores.items():
            keep = group & (common if common_only else np.isfinite(s))
            yy, ss = y[keep], s[keep]
            both = len(np.unique(yy)) == 2
            auc = float(roc_auc_score(yy, ss)) if both else np.nan
            ap = float(average_precision_score(yy, ss)) if both else np.nan
            for policy, threshold in zip(('alert', 'strict'), thresholds[name]):
                positive = ss > threshold
                tp = int((positive & (yy == 1)).sum())
                fp = int((positive & (yy == 0)).sum())
                na, nb = int((yy == 1).sum()), int((yy == 0).sum())
                rows.append(dict(variant=variant, method=name, policy=policy,
                    total_windows=int(group.sum()), scored_windows=int(keep.sum()),
                    excluded_windows=int(group.sum()-keep.sum()),
                    benign_windows=nb, attack_windows=na, roc_auc=auc, average_precision=ap,
                    threshold=threshold, attack_recall_pct=100*tp/na if na else np.nan,
                    benign_fpr_pct=100*fp/nb if nb else np.nan,
                    precision_pct=100*tp/(tp+fp) if tp+fp else np.nan,
                    f1=2*tp/(2*tp+fp+na-tp) if 2*tp+fp+na-tp else np.nan))
    return pd.DataFrame(rows)


def run(args):
    from fedcontext.inference import load_detector
    model, _, run_info = load_detector(args.run)
    if Path(args.out).exists():
        raise FileExistsError(f'{args.out} exists; choose a NEW output name, e.g. speed_checks_v2.')
    print('Loading benign reference/calibration windows...', flush=True)
    d, meta, info = load(args.calibration_prepared, run_info)
    ri, ci = select(meta, 'score_ref', True), select(meta, 'calibration', True)
    minimum = int(run_info['config'].get('calibration_min_windows', 200))
    if min(len(ri), len(ci)) < minimum:
        raise ValueError(f'Need at least {minimum} benign windows in each calibration role.')
    ref_sessions = set(meta.iloc[ri].session.astype(str))
    cal_sessions = set(meta.iloc[ci].session.astype(str))
    if ref_sessions & cal_sessions:
        raise ValueError('Reference and calibration sessions overlap.')
    print(f'Scoring reference={len(ri)}, calibration={len(ci)} with frozen weights...', flush=True)
    reference = compute(model, d, ri, info, args.batch_size)
    calibration = compute(model, d, ci, info, args.batch_size)
    # Positive scale from benign reference only. Scores remain unbounded, avoiding
    # empirical-percentile saturation when several errors exceed the reference max.
    raw_scales = np.quantile(reference['errors'], .95, axis=0, method='higher')
    feature_scales = np.maximum(raw_scales, 1e-12)
    residuals = reference['residual'][np.isfinite(reference['residual'])]
    if len(residuals) < minimum:
        raise ValueError('Too few valid reference timestamp intervals for consistency calibration.')
    endpoint_ref = reference['endpoint_residual']
    endpoint_ref = endpoint_ref[np.isfinite(endpoint_ref)]
    physics_scale = [max(float(np.quantile(r, .95, method='higher')), 1e-6)
                     for r in (residuals, endpoint_ref)]
    q_low, q_high = float(run_info['config']['q_low']), float(run_info['config']['q_high'])
    if not 0 < q_low < q_high < 1:
        raise ValueError('Invalid quantile settings in saved run.')
    cal_scores = make_scores(calibration, feature_scales, physics_scale)
    thresholds, threshold_rows = {}, []
    for name, s in cal_scores.items():
        valid = s[np.isfinite(s)]
        if len(valid) < minimum:
            raise ValueError(f'Too few valid calibration observations for {name}.')
        low, high = np.quantile(valid, [q_low, q_high], method='higher')
        thresholds[name] = [float(low), float(high)]
        threshold_rows.append(dict(method=name, calibration_count=len(valid), low=low, high=high,
            calibration_alert_pct=100*np.mean(valid > low),
            calibration_strict_pct=100*np.mean(valid > high)))
    del d, meta
    print('Loading and scoring TEST windows...', flush=True)
    td, tm, ti = load(args.test_prepared, run_info)
    ids = select(tm, 'test')
    target_meta = tm.iloc[ids].reset_index(drop=True)
    if set(target_meta.session.astype(str)) & (ref_sessions | cal_sessions):
        raise ValueError('Test sessions overlap calibration sessions; check official splits.')
    result = compute(model, td, ids, ti, args.batch_size)
    scores = make_scores(result, feature_scales, physics_scale)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=False)
    pred = target_meta.copy()
    for i, name in enumerate(FEATURES):
        pred[f'actual_{name}'] = result['actual'][:, i]
        pred[f'predicted_{name}'] = result['pred'][:, i]
        pred[f'squared_error_{name}_normalized'] = result['errors'][:, i]
    for k in ('dt', 'previous_speed', 'estimated', 'avg_reported', 'residual',
              'endpoint_estimated', 'endpoint_residual'):
        pred[k] = result[k]
    pred['consistency_valid'] = np.isfinite(result['residual'])
    for name, s in scores.items():
        low, high = thresholds[name]
        pred[f'score_{name}'] = s
        pred[f'decision_{name}'] = np.where(~np.isfinite(s), 'Uncertain',
            np.where(s > high, 'Suspicious', np.where(s > low, 'Uncertain', 'Trusted')))
    pred.to_csv(output / 'message_diagnostics.csv.gz', index=False)
    metrics = evaluate(target_meta, scores, thresholds)
    common = evaluate(target_meta, scores, thresholds, True)
    metrics.to_csv(output / 'metrics.csv', index=False)
    common.to_csv(output / 'metrics_common_windows.csv', index=False)
    pd.DataFrame(threshold_rows).to_csv(output / 'thresholds.csv', index=False)
    settings = dict(run=str(Path(args.run).resolve()),
        model_sha256=hashlib.sha256((Path(args.run)/'model.pt').read_bytes()).hexdigest(),
        calibration_prepared=str(Path(args.calibration_prepared).resolve()),
        test_prepared=str(Path(args.test_prepared).resolve()),
        features=FEATURES, error_reference_q95=feature_scales.tolist(),
        near_zero_reference_features=[FEATURES[i] for i in np.flatnonzero(raw_scales < 1e-12)],
        consistency_reference_q95_mps=physics_scale, thresholds=thresholds,
        q_low=q_low, q_high=q_high, score_ref_windows=len(ri), calibration_windows=len(ci),
        test_windows=len(ids), invalid_consistency_test_windows=int((~pred.consistency_valid).sum()),
        method='Benign Q95 scaling followed by max; separate benign quantile thresholds.',
        note='Frozen-model scoring experiment. No training or deployment changes. Test labels only for metrics.')
    (output / 'diagnostic_calibration.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')
    methods = ['movement_original_mean', 'error_speed', 'movement_calibrated_max',
               'speed_position_consistency', 'speed_position_endpoint']
    view = common[common.method.isin(methods) & common.policy.eq('strict')]
    print('\nSTRICT RESULTS ON THE SAME VALID WINDOWS (fair comparison):')
    print(view[['variant', 'method', 'roc_auc', 'attack_recall_pct', 'benign_fpr_pct',
                'excluded_windows']].to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print('\nSaved:', output.resolve())
    print('Timing/context streams and the original deployed detector were not modified.')
    print('Invalid consistency intervals remain Uncertain and are excluded from score metrics; see coverage counts.')
    if settings['near_zero_reference_features']:
        print('WARNING: near-zero reference errors for:', settings['near_zero_reference_features'])
        print('Inspect these features before accepting the combined score.')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--list-prepared', metavar='DIRECTORY')
    p.add_argument('--run', default='runs/federated_final_v1')
    p.add_argument('--calibration-prepared', help='Original prepared folder with benign score_ref and calibration.')
    p.add_argument('--test-prepared', help='Prepared folder containing all attack variants to evaluate.')
    p.add_argument('--out', default='runs/speed_checks_v1')
    p.add_argument('--batch-size', type=int, default=512)
    args = p.parse_args()
    if args.list_prepared:
        return list_prepared(args.list_prepared)
    if not args.calibration_prepared or not args.test_prepared:
        p.error('--calibration-prepared and --test-prepared are required; use --list-prepared data to find them.')
    if args.batch_size < 1:
        p.error('--batch-size must be positive.')
    run(args)


if __name__ == '__main__':
    main()
