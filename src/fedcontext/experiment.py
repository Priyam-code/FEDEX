from pathlib import Path
import hashlib
import platform
import time
import importlib.metadata
import numpy as np
import pandas as pd
import torch
from .prepare import load_prepared
from .training import fit, score, KEYS
from .calibration import Calibrator, SUBSETS, subset_name
from .metrics import evaluate, episode_delays
from .baselines import fit_baselines
from .reporting import plots, summarize
from .utils import dump_json, fresh_dir, save_csv


def evaluate_saved(prepared, checkpoint, output):
    """New held-out data may be scored without fitting anything on it."""
    from .inference import load_detector
    model, cal, run_info = load_detector(checkpoint)
    data, meta, info = load_prepared(prepared)
    cfg = run_info["config"]
    for k in ("history", "max_gap_s", "neighbor_ttl_s", "neighbor_radius_m"):
        if cfg[k] != info["config"][k]:
            raise ValueError(f"Feature mismatch: {k}")
    if info["road_map"] != run_info["road_map"]:
        raise ValueError("Road-map mode mismatch")
    ids = np.flatnonzero(meta.partition.eq("test"))
    if not len(ids):
        raise ValueError("No test windows to evaluate")
    out = fresh_dir(output)
    errors = score(model, data, ids)
    export_evaluation(out, meta.iloc[ids].reset_index(drop=True), errors, cal,
                      run_info["method"], run_info["seed"], info["origin"])
    dump_json(out/"evaluation.json", dict(origin=info["origin"], checkpoint=str(Path(checkpoint).resolve()),
        checkpoint_origin=run_info["origin"], seed=run_info["seed"], config=cfg,
        note="Existing weights and thresholds; no refit on this dataset."))
    dump_json(out/"latency.json", benchmark(model, data, ids, cal, cfg["latency_samples"]))
    raw = pd.read_csv(Path(prepared)/"messages.csv.gz", usecols=["split", "variant", "label"])
    counts = []
    target = meta.iloc[ids]
    for (variant, lab), g in raw[raw.split=="test"].groupby(["variant", "label"]):
        count = int(((target.variant==variant)&(target.label==lab)).sum())
        counts.append(dict(variant=variant, label=int(lab), raw_messages=len(g), scored_windows=count, coverage=count/len(g)))
    dump_json(out/"coverage.json", dict(groups=counts))
    return out


def benchmark(model, data, ids, calibrator, count):
    model.eval()
    times = []
    indices = ids[:min(count, len(ids))]
    with torch.inference_mode():
        for iteration in range(10+len(indices)):
            idx = indices[iteration % len(indices)]
            batch = [torch.from_numpy(data[k][idx:idx+1]) for k in KEYS]
            start = time.perf_counter_ns()
            errors = model.errors(*batch).numpy()
            calibrator.predict(errors)
            duration = (time.perf_counter_ns()-start)/1e6
            if iteration >= 10:
                times.append(duration)
    return dict(device="CPU", batch_size=1, samples=len(times), median_ms=float(np.median(times)),
                p95_ms=float(np.quantile(times, .95)), mean_ms=float(np.mean(times)),
                scope="Three neural forward passes + error computation + calibration/fusion. Excludes JSON, features, networking, warmup history and actuation.",
                hardware=platform.platform(), processor=platform.processor(), torch_threads=torch.get_num_threads())


def export_evaluation(out, test_meta, errors, calibrator, method, seed, origin):
    rows, family = [], []
    for indices in SUBSETS:
        scores, decisions, _ = calibrator.predict(errors, indices)
        for policy in ("strict", "alert"):
            row = evaluate(test_meta.label, scores, decisions, policy)
            row.update(method=method, seed=seed, subset=subset_name(indices))
            rows.append(row)
    scores, decisions, ranked = calibrator.predict(errors)
    pred = test_meta.reset_index(drop=True).copy()
    pred["score"], pred["decision"] = scores, decisions
    for i, stream in enumerate(("movement", "timing", "context")):
        pred[f"{stream}_mse"] = errors[:, i]
        pred[f"{stream}_rank"] = ranked[:, i]
    save_csv(pred, out/"predictions.csv.gz")
    for (scenario, attack), group in pred.groupby(["scenario", "variant"]):
        for policy in ("strict", "alert"):
            family.append(dict(scenario=scenario, attack=attack, method=method, seed=seed,
                               **evaluate(group.label, group.score, group.decision, policy)))
    pd.DataFrame(family).to_csv(out/"per_scenario_attack.csv", index=False)
    episodes = episode_delays(pred)
    pd.DataFrame(episodes, columns=["session", "pseudonym", "variant", "first_scored_attack_time", "scored_attack_windows", "detected", "delay_s"]).to_csv(out/"episode_delays.csv", index=False)
    dump_json(out/"episode_summary.json", dict(episodes=len(episodes),
        detected=sum(e["detected"] for e in episodes),
        note="Latency measured from FIRST SCORED attack message. Undetected episodes are retained with null delay, not averaged as zero. Pseudonym changes/warmup obscure actual onset."))
    pd.DataFrame(rows).to_csv(out/"metrics.csv", index=False)
    plots(out, pred, rows, origin)
    return rows


def run(prepared, output, cfg, modes, seeds, baselines=True):
    out = fresh_dir(output)
    data, meta, info = load_prepared(prepared)
    # Preparation settings define feature semantics; do not silently change them later.
    for k in ("history", "clients", "partition", "region_width_m", "max_gap_s", "neighbor_ttl_s", "neighbor_radius_m"):
        if cfg[k] != info["config"][k]:
            raise ValueError(f"Config {k} differs from prepared data; re-run prepare")
    groups = {p: np.flatnonzero(meta.partition.eq(p)) for p in ("train", "selection", "score_ref", "calibration", "test")}
    for p, ids in groups.items():
        required = cfg["calibration_min_windows"] if p in ("score_ref", "calibration") else 1
        if len(ids) < required:
            raise ValueError(f"{p} has {len(ids)} windows; requires {required}. Add receiver files, do not borrow test data.")
    for p in ("train", "selection", "score_ref", "calibration"):
        if meta.iloc[groups[p]].label.ne(0).any() or not meta.iloc[groups[p]].clean_window.all():
            raise ValueError("Nonbenign fitting/calibration partition")
    env = {m: importlib.metadata.version(m) for m in ("numpy", "pandas", "torch", "scikit-learn", "matplotlib", "ijson")}
    test_meta = meta.iloc[groups["test"]].reset_index(drop=True)
    prepared_hash = hashlib.sha256(Path(prepared, "windows.csv").read_bytes()).hexdigest()
    dump_json(out/"experiment.json", dict(origin=info["origin"], seeds=seeds, modes=modes, config=cfg,
        environment=env, python=platform.python_version(), prepared_metadata_sha256=prepared_hash))
    for seed in seeds:
        for mode in modes:
            destination = fresh_dir(out/f"{mode}_seed{seed}")
            dump_json(destination/"run.json", dict(origin=info["origin"], seed=seed, method=mode, config=cfg, environment=env,
                road_map=info["road_map"], prepared_metadata_sha256=prepared_hash))
            model, stats = fit(data, meta, cfg, mode, seed, destination)
            errors = score(model, data)
            calibrator = Calibrator().fit(errors[groups["score_ref"]], errors[groups["calibration"]], cfg["q_low"], cfg["q_high"])
            dump_json(destination/"calibration.json", calibrator.to_dict())
            save_csv(pd.DataFrame({"record_id":meta.record_id, "partition":meta.partition,
                          "movement_mse":errors[:, 0], "timing_mse":errors[:, 1], "context_mse":errors[:, 2]}), destination/"all_errors.csv.gz")
            export_evaluation(destination, test_meta, errors[groups["test"]], calibrator, mode, seed, info["origin"])
            dump_json(destination/"latency.json", benchmark(model, data, groups["test"], calibrator, cfg["latency_samples"]))
            raw = pd.read_csv(Path(prepared)/"messages.csv.gz", usecols=["split", "variant", "label"])
            counts = []
            for (variant, lab), g in raw[raw.split=="test"].groupby(["variant", "label"]):
                scored = int(((test_meta.variant==variant)&(test_meta.label==lab)).sum())
                counts.append(dict(variant=variant, label=int(lab), raw_messages=len(g), scored_windows=scored, coverage=scored/len(g)))
            dump_json(destination/"coverage.json", dict(groups=counts, average_feature_ms=info["average_feature_ms"],
                note="Warm-up/reset histories are excluded; coverage must accompany detection rates."))
        if baselines:
            destination = fresh_dir(out/f"baselines_seed{seed}")
            results = fit_baselines(data, meta, cfg, seed, destination)
            rows, family, settings = [], [], {}
            for name, result in results.items():
                scores, decisions = result["scores"][groups["test"]], result["decisions"][groups["test"]]
                pred = test_meta.copy()
                pred["score"], pred["decision"] = scores, decisions
                save_csv(pred, destination/(name+"_predictions.csv.gz"))
                for policy in ("strict", "alert"):
                    rows.append(dict(method=name, seed=seed, subset="all_features", **evaluate(pred.label, scores, decisions, policy)))
                for (scenario, attack), g in pred.groupby(["scenario", "variant"]):
                    for policy in ("strict", "alert"):
                        family.append(dict(scenario=scenario, attack=attack, method=name, seed=seed, **evaluate(g.label, g.score, g.decision, policy)))
                settings[name] = {k: result[k] for k in ("train_samples", "low", "high")}
            pd.DataFrame(rows).to_csv(destination/"metrics.csv", index=False)
            pd.DataFrame(family).to_csv(destination/"per_scenario_attack.csv", index=False)
            dump_json(destination/"run.json", dict(origin=info["origin"], seed=seed, config=cfg, settings=settings))
    summarize(out)
    return out
