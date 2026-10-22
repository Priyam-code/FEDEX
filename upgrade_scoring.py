"""Additive scoring upgrade for original FedContextTrust_V2X v1.

Save in the project root. No training changes or weight updates.
Commands: calibrate, evaluate, predict-example. See --help and README.
For live messages: from upgrade_scoring import StreamingDetector.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parent
if (ROOT / 'src' / 'fedcontext').is_dir():
    sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import pandas as pd
import torch
from fedcontext.inference import load_detector
from fedcontext.prepare import load_prepared
from fedcontext.features import ReceiverFeatures, RoadMap

NAMES = ['movement_dx', 'movement_dy', 'movement_speed', 'movement_acceleration',
         'movement_heading_sin', 'movement_heading_cos', 'timing_mse', 'context_mse',
         'speed_position_endpoint']
FEATURES = ['dx', 'dy', 'speed', 'acceleration', 'heading_sin', 'heading_cos']
SCALES = np.array([10., 10., 40., 10., 1., 1.])
PROFILE_SIZE = {'featurewise': 8, 'hybrid': 9}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_prepared(info, run_info):
    if info['features']['movement'] != FEATURES:
        raise ValueError('This upgrade requires original v1 six-feature movement inputs.')
    if not np.allclose(info['fixed_scales']['movement'], SCALES):
        raise ValueError('Unexpected movement scaling; do not mix feature versions.')
    if info['features']['timing'][1] != 'send_gap' or info['fixed_scales']['timing'][1] != 1:
        raise ValueError('Expected send_gap in seconds at timing index 1.')
    for k in ('history', 'max_gap_s', 'neighbor_ttl_s', 'neighbor_radius_m'):
        if info['config'][k] != run_info['config'][k]:
            raise ValueError(f'Feature configuration mismatch: {k}')
    if info['road_map'] != run_info['road_map']:
        raise ValueError('Road map mode mismatch.')


def partition(meta, name, clean=False):
    ids = np.flatnonzero(meta.partition.eq(name))
    if not len(ids):
        raise ValueError(f'No {name} windows in this prepared folder.')
    if clean:
        selected = meta.iloc[ids]
        clean_mask = selected.clean_window.astype(str).str.lower().isin(['true', '1'])
        if not selected.label.eq(0).all() or not clean_mask.all():
            raise ValueError(f'{name} contains non-benign/unclean windows; refusing fit.')
    return ids


@torch.inference_mode()
def evidence(model, movement, target, timing, context, cfg):
    """Same function for prepared batches and OBU streaming; original weights."""
    tensors = [torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
               for a in (movement, target, timing, context)]
    m, y, t, c = tensors
    movement_squared = (model.movement(m) - y).square()
    m_error = movement_squared.cpu().numpy().astype(float)
    t_error = (model.timing(t) - t).square().mean(dim=(1, 2)).cpu().numpy()
    c_error = (model.context(c) - c).square().mean(dim=1).cpu().numpy()
    # Target displacement and current send_gap span the same sender interval.
    current = np.asarray(target, dtype=float) * SCALES
    dt = np.asarray(timing[:, -1, 1], dtype=float)
    valid = np.isfinite(dt) & (dt > 1e-6) & (dt <= cfg['max_gap_s'])
    endpoint_error = np.full(len(y), np.nan)
    estimate = np.hypot(current[valid, 0], current[valid, 1]) / dt[valid]
    estimate += 0.5 * current[valid, 3] * dt[valid]
    endpoint_error[valid] = np.abs(current[valid, 2] - estimate)
    raw = np.column_stack([m_error, t_error, c_error, endpoint_error])
    if not np.isfinite(raw[:, :8]).all():
        raise ValueError('Non-finite neural model errors.')
    # Keep original torch float32 reduction for the legacy comparison.
    old_m = movement_squared.mean(dim=1).cpu().numpy()
    old = np.column_stack([old_m, t_error, c_error])
    return raw, old


def score_windows(model, data, ids, cfg, batch_size):
    raw, old = [], []
    for start in range(0, len(ids), batch_size):
        selected = ids[start:start+batch_size]
        r, o = evidence(model, *(data[k][selected] for k in ('movement', 'target', 'timing', 'context')), cfg)
        raw.append(r)
        old.append(o)
    return np.concatenate(raw), np.concatenate(old)


class EvidenceCalibration:
    """Benign empirical-tail transform, max fusion, disjoint threshold calibration.

    Evidence is NOT an attack probability. Empirical upper-tail rarity is
    -log((n+1-count(reference<error))/(n+1)). Above the largest reference error,
    log1p excess extends the score to avoid a flat percentile ceiling. The
    extension width is max(Q99-Q95, 0.01*abs(max_reference), 1e-12).
    All choices are fixed before test scoring. Separate profile thresholds are
    fitted on benign calibration windows; no test labels enter this class.
    """
    def __init__(self, obj):
        self.obj = obj
        self.references = [np.array(r, dtype=float) for r in obj['references']]
        self.widths = np.array(obj['extension_widths'], dtype=float)
        self.thresholds = obj['thresholds']

    def transform(self, raw):
        transformed = np.full(raw.shape, np.nan)
        for j, ref in enumerate(self.references):
            valid = np.isfinite(raw[:, j])
            x = raw[valid, j]
            below = np.searchsorted(ref, x, side='left')
            value = -np.log((len(ref)+1-below)/(len(ref)+1))
            extra = np.maximum(x-ref[-1], 0)/self.widths[j]
            value += np.log1p(extra)
            transformed[valid, j] = value
        return transformed

    def predict(self, raw, profile='hybrid'):
        transformed = self.transform(raw)
        count = PROFILE_SIZE[profile]
        valid = np.isfinite(transformed[:, :count]).all(axis=1)
        values = np.max(transformed[:, :count], axis=1)
        low, high = self.thresholds[profile]
        decisions = np.where(~valid, 'Uncertain',
                    np.where(values > high, 'Suspicious',
                    np.where(values > low, 'Uncertain', 'Trusted')))
        return values, decisions, transformed

    @classmethod
    def fit(cls, reference, calibration, q_low, q_high, minimum):
        refs, widths = [], []
        for j in range(reference.shape[1]):
            r = np.sort(reference[np.isfinite(reference[:, j]), j])
            if len(r) < minimum:
                raise ValueError(f'Too few benign reference windows for {NAMES[j]}.')
            refs.append(r.tolist())
            q95, q99 = np.quantile(r, [.95, .99], method='higher')
            widths.append(float(max(q99-q95, .01*abs(r[-1]), 1e-12)))
        obj = dict(references=refs, extension_widths=widths, thresholds={}, names=NAMES,
                   transform='empirical_upper_tail_with_log_excess_v1', q_low=q_low, q_high=q_high)
        fitted = cls(obj)
        transformed = fitted.transform(calibration)
        for profile, count in PROFILE_SIZE.items():
            valid = np.isfinite(transformed[:, :count]).all(axis=1)
            if valid.sum() < minimum:
                raise ValueError(f'Too few benign calibration windows for {profile}.')
            fused = transformed[valid, :count].max(axis=1)
            obj['thresholds'][profile] = np.quantile(fused, [q_low, q_high], method='higher').tolist()
        return cls(obj)


def calibrate(args):
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f'{out} already exists. Choose a new output folder.')
    model, _, run_info = load_detector(args.run)
    data, meta, info = load_prepared(args.prepared)
    check_prepared(info, run_info)
    ri, ci = partition(meta, 'score_ref', True), partition(meta, 'calibration', True)
    rs, cs = set(meta.iloc[ri].session.astype(str)), set(meta.iloc[ci].session.astype(str))
    if rs & cs:
        raise ValueError('Reference and calibration sessions overlap.')
    minimum = int(run_info['config'].get('calibration_min_windows', 200))
    print(f'Frozen weights: scoring {len(ri)} reference + {len(ci)} benign calibration windows...', flush=True)
    ref, _ = score_windows(model, data, ri, run_info['config'], args.batch_size)
    cal, _ = score_windows(model, data, ci, run_info['config'], args.batch_size)
    ql, qh = run_info['config']['q_low'], run_info['config']['q_high']
    new = EvidenceCalibration.fit(ref, cal, ql, qh, minimum)
    obj = new.obj | dict(version=1, config=run_info['config'], road_map=info['road_map'],
        model_sha256=sha(Path(args.run)/'model.pt'), source_run=str(Path(args.run).resolve()),
        calibration_prepared=str(Path(args.prepared).resolve()),
        fitting_sessions=sorted(rs | cs), reference_count=len(ri), calibration_count=len(ci),
        note='Frozen-model scoring upgrade; all fits benign-only; not a new training run.')
    out.mkdir(parents=True, exist_ok=False)
    for filename in ('model.pt', 'run.json', 'calibration.json'):
        shutil.copy2(Path(args.run)/filename, out/filename)
    if (Path(args.prepared)/'road_map.json').exists():
        shutil.copy2(Path(args.prepared)/'road_map.json', out/'road_map.json')
    write(out/'scoring_upgrade.json', obj)
    rows = []
    for profile in PROFILE_SIZE:
        values, decisions, _ = new.predict(cal, profile)
        valid = np.isfinite(values)
        rows.append(dict(profile=profile, low=new.thresholds[profile][0], high=new.thresholds[profile][1],
            calibration_valid=int(valid.sum()), calibration_missing=int((~valid).sum()),
            calibration_strict_fpr_pct=100*float(np.mean(decisions[valid]=='Suspicious')),
            calibration_alert_fpr_pct=100*float(np.mean(decisions[valid]!='Trusted'))))
    pd.DataFrame(rows).to_csv(out/'scoring_thresholds.csv', index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print('Saved upgraded scoring artifact:', out)
    print('Original weights/calibration unchanged. Use THIS script or its StreamingDetector for upgraded scoring.')


def load_upgrade(run):
    model, legacy, info = load_detector(run)
    obj = read(Path(run)/'scoring_upgrade.json')
    if obj['model_sha256'] != sha(Path(run)/'model.pt'):
        raise ValueError('Model weights changed since scoring calibration. Recalibrate.')
    if obj['config'] != info['config'] or obj['road_map'] != info['road_map']:
        raise ValueError('Saved upgrade configuration mismatch.')
    return model, legacy, info, EvidenceCalibration(obj)


def metrics(meta, profiles, common):
    from sklearn.metrics import roc_auc_score, average_precision_score
    rows = []
    y = meta.label.to_numpy(dtype=int)
    groups = [('ALL', np.ones(len(meta), dtype=bool))]
    groups += [(str(v), meta.variant.eq(v).to_numpy()) for v in sorted(meta.variant.unique())]
    for variant, group in groups:
        for name, (score, decision) in profiles.items():
            valid = group & common
            yy, ss, dd = y[valid], score[valid], decision[valid]
            both = len(np.unique(yy)) == 2
            auc = float(roc_auc_score(yy, ss)) if both else np.nan
            ap = float(average_precision_score(yy, ss)) if both else np.nan
            for policy in ('strict', 'alert'):
                positive = dd == 'Suspicious' if policy == 'strict' else dd != 'Trusted'
                tp, fp = int(np.sum(positive & (yy==1))), int(np.sum(positive & (yy==0)))
                na, nb = int(np.sum(yy==1)), int(np.sum(yy==0))
                rows.append(dict(variant=variant, method=name, policy=policy, roc_auc=auc,
                    average_precision=ap, attack_recall_pct=100*tp/na if na else np.nan,
                    benign_fpr_pct=100*fp/nb if nb else np.nan,
                    precision_pct=100*tp/(tp+fp) if tp+fp else np.nan,
                    f1=2*tp/(tp+fp+na) if tp+fp+na else np.nan,
                    attack_windows=na, benign_windows=nb, excluded_windows=int(group.sum()-valid.sum())))
    return pd.DataFrame(rows)


def make_plots(out, meta, profiles, common):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
    import re
    figure_dir = out/'figures'
    figure_dir.mkdir()
    for variant in ['ALL'] + sorted(meta.variant.unique()):
        mask = common & (np.ones(len(meta), dtype=bool) if variant=='ALL' else meta.variant.eq(variant).to_numpy())
        y = meta.loc[mask, 'label'].to_numpy(dtype=int)
        if len(np.unique(y)) != 2:
            continue
        slug = re.sub(r'[^A-Za-z0-9_-]', '_', str(variant))
        for kind in ('ROC', 'PR'):
            fig, ax = plt.subplots(figsize=(6.8, 5.3))
            for name, (s, _) in profiles.items():
                scores = s[mask]
                if kind=='ROC':
                    x, z, _ = roc_curve(y, scores)
                    label = f'{name} (AUC {roc_auc_score(y, scores):.4f})'
                else:
                    z, x, _ = precision_recall_curve(y, scores)
                    label = f'{name} (AP {average_precision_score(y, scores):.4f})'
                ax.plot(x, z, label=label, linewidth=1.8)
            if kind=='ROC':
                ax.plot([0,1], [0,1], '--', color='grey', linewidth=1)
                ax.set(xlabel='False positive rate', ylabel='True positive rate (recall)')
            else:
                ax.axhline(y.mean(), linestyle='--', color='grey', linewidth=1)
                ax.set(xlabel='Recall', ylabel='Precision')
            ax.set(title=f'{variant}: {kind} comparison', xlim=(0,1), ylim=(0,1.02))
            ax.grid(alpha=.2)
            ax.legend(fontsize=8, loc='best')
            fig.tight_layout()
            fig.savefig(figure_dir/f'{slug}_{kind}.png', dpi=300)
            fig.savefig(figure_dir/f'{slug}_{kind}.pdf')
            plt.close(fig)


def evaluate(args):
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f'{out} already exists; choose a new output folder.')
    model, legacy, info, new = load_upgrade(args.run)
    data, meta, prep = load_prepared(args.prepared)
    check_prepared(prep, info)
    ids = partition(meta, 'test')
    test = meta.iloc[ids].reset_index(drop=True)
    if set(test.session.astype(str)) & set(new.obj['fitting_sessions']):
        raise ValueError('Test sessions overlap fitting sessions.')
    print(f'Scoring {len(ids)} test windows; no fitting on Test...', flush=True)
    raw, old = score_windows(model, data, ids, info['config'], args.batch_size)
    profiles = {}
    s, d, _ = legacy.predict(old)
    profiles['original_fusion'] = (s, d)
    s, d, _ = legacy.predict(old, (0,))
    # Raw movement MSE preserves ranking; decisions retain original thresholds.
    profiles['original_movement'] = (old[:,0], d)
    for profile in PROFILE_SIZE:
        s, d, transformed = new.predict(raw, profile)
        profiles[profile] = (s, d)
    common = np.logical_and.reduce([np.isfinite(s) for s, _ in profiles.values()])
    out.mkdir(parents=True, exist_ok=False)
    result = test.copy()
    for j, name in enumerate(NAMES):
        result[f'raw_{name}'] = raw[:, j]
        result[f'evidence_{name}'] = transformed[:, j]
    for profile, (s, d) in profiles.items():
        result[f'{profile}_score'], result[f'{profile}_decision'] = s, d
    result['score'], result['decision'] = profiles[args.profile]
    result['common_valid'] = common
    result.to_csv(out/'predictions.csv.gz', index=False)
    table = metrics(test, profiles, common)
    table.to_csv(out/'metrics_comparison.csv', index=False)
    # Separate all-window decision counts retain invalid intervals as Uncertain.
    counts = []
    for profile, (_, decisions) in profiles.items():
        frame = test[['variant','label']].copy()
        frame['decision'] = decisions
        frame['method'] = profile
        counts.append(frame.groupby(['variant','label','method','decision']).size().rename('windows').reset_index())
    pd.concat(counts).to_csv(out/'decision_counts_all_windows.csv', index=False)
    write(out/'evaluation.json', dict(prepared=str(Path(args.prepared).resolve()),
        run=str(Path(args.run).resolve()), deployed_profile=args.profile,
        model_sha256=new.obj['model_sha256'], scoring_sha256=sha(Path(args.run)/'scoring_upgrade.json'),
        test_windows=len(ids), common_valid_windows=int(common.sum()),
        excluded_windows=int((~common).sum()),
        note='Comparison curves/metrics use identical valid windows. Invalid hybrid intervals remain Uncertain in predictions. Development evaluation; no attack-label fitting.'))
    make_plots(out, test, profiles, common)
    print(table[table.policy.eq('strict')][['variant','method','roc_auc','attack_recall_pct','benign_fpr_pct']].to_string(index=False))
    print('Saved predictions, metrics, and PNG/PDF figures in:', out)


class StreamingDetector:
    """Use one instance per OBU receiver/session. No online weight updates."""
    def __init__(self, run, road_path=None, profile='hybrid'):
        if profile not in PROFILE_SIZE:
            raise ValueError(f'Choose profile from {list(PROFILE_SIZE)}')
        self.model, _, self.info, self.cal = load_upgrade(run)
        self.profile = profile
        if bool(road_path) != self.info['road_map']:
            raise ValueError('Use the road-map mode of the saved model.')
        self.engine = ReceiverFeatures(self.info['config'], RoadMap(road_path))

    def update(self, observation):
        keys = ('rx_time','tx_time','pseudonym','message_id','x','y','rx_x','rx_y','speed','acceleration','heading')
        row = {k: observation[k] for k in keys}
        numeric = [row[k] for k in keys if k not in ('pseudonym','message_id')]
        if not np.isfinite(np.asarray(numeric, dtype=float)).all():
            raise ValueError('Non-finite observable message fields.')
        features = self.engine.update(row)
        if features is None:
            return dict(decision='Uncertain', score=None, reason='insufficient_history')
        raw, old = evidence(self.model, *(features[k][None] for k in ('movement','target','timing','context')), self.info['config'])
        values, decisions, transformed = self.cal.predict(raw, self.profile)
        finite = np.isfinite(values[0])
        safe = lambda a: [float(x) if np.isfinite(x) else None for x in a]
        return dict(decision=str(decisions[0]), score=float(values[0]) if finite else None,
                    reason='model_score' if finite else 'invalid_consistency_interval',
                    profile=self.profile, evidence_names=NAMES, raw_errors=safe(raw[0]),
                    calibrated_evidence=safe(transformed[0]), stream_mse=safe(old[0]))


def predict_example(args):
    model_info = read(Path(args.run)/'run.json')
    _, meta, prep = load_prepared(args.prepared)
    check_prepared(prep, model_info)
    targets = meta[meta.partition.eq('test') & meta.variant.eq(args.variant) & meta.label.eq(args.label)]
    if targets.empty:
        raise ValueError('No eligible test window for the requested variant/label.')
    selected = targets.iloc[0]
    messages = pd.read_csv(Path(args.prepared)/'messages.csv.gz',
        dtype={k:str for k in ('record_id','session','receiver','pseudonym','message_id')})
    session = messages[messages.session.eq(str(selected.session))].sort_values('rx_time', kind='stable').reset_index(drop=True)
    matches = np.flatnonzero(session.record_id.eq(str(selected.record_id)))
    if len(matches) != 1:
        raise ValueError('Target record does not match exactly one message.')
    detector = StreamingDetector(args.run, road_path=args.road, profile=args.profile)
    position = int(matches[0])
    for row in session.iloc[:position].to_dict('records'):
        detector.update(row)
    message = session.iloc[position].to_dict()
    result = detector.update(message)
    print(json.dumps(dict(receiver=message['receiver'], sender=message['pseudonym'],
        message_id=message['message_id'], result=result), indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('calibrate','evaluate','predict-example'):
        p = sub.add_parser(name)
        p.add_argument('--prepared', required=True)
        p.add_argument('--run', required=True)
        p.add_argument('--batch-size', type=int, default=512)
        if name != 'predict-example':
            p.add_argument('--out', required=True)
        if name != 'calibrate':
            p.add_argument('--profile', choices=PROFILE_SIZE, default='hybrid')
        if name=='predict-example':
            p.add_argument('--variant', default='randomSpeedOffset')
            p.add_argument('--label', type=int, choices=(0,1), default=1)
            p.add_argument('--road')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be positive.')
    {'calibrate':calibrate, 'evaluate':evaluate, 'predict-example':predict_example}[args.command](args)


if __name__=='__main__':
    main()
