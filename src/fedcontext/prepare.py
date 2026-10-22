from pathlib import Path
import shutil
import time
import numpy as np
import pandas as pd
from .features import ReceiverFeatures, RoadMap, client_id, NAMES, SCALES
from .ingest import ingest
from .utils import dump_json, fresh_dir, read_json, stable_hash


def prepare(manifest, output, cfg, road_path=None):
    out = fresh_dir(output)
    frame, audit = ingest(manifest, out)
    road = RoadMap(road_path)
    if road_path:
        shutil.copyfile(road_path, out/"road_map.json")
    arrays = {k: [] for k in ("movement", "target", "timing", "context")}
    meta, counts = [], {}
    elapsed = 0.
    for session, group in frame.groupby("session", sort=True):
        engine = ReceiverFeatures(cfg, road)
        for row in group.sort_values("rx_time", kind="stable").to_dict("records"):
            start = time.perf_counter()
            f = engine.update(row)
            elapsed += time.perf_counter()-start
            if f is None:
                continue
            if row["split"] != "test" and not f["clean_window"]:
                continue
            part = row["split"]
            if part == "validation":
                # Whole receiver files go to one of three disjoint roles.
                # Selection chooses epochs; score_ref fits marginal ECDF;
                # calibration selects fusion thresholds. Test never participates.
                bucket = stable_hash(f"{row['scenario']}:{row['receiver']}") % 3
                part = ("selection", "score_ref", "calibration")[bucket]
            for k in arrays:
                arrays[k].append(f[k])
            keep = {k: row[k] for k in ["record_id", "session", "receiver", "scenario", "variant", "pseudonym", "message_id", "rx_time", "label", "source_file"]}
            keep.update(partition=part, client=client_id(row, cfg), clean_window=f["clean_window"])
            meta.append(keep)
            counts[part] = counts.get(part, 0)+1
    if not meta:
        raise ValueError("No complete histories. Check timestamps, history length and sender_alias.")
    np.savez_compressed(out/"windows.npz", **{k: np.asarray(v, dtype=np.float32) for k, v in arrays.items()})
    metadata = pd.DataFrame(meta)
    metadata.to_csv(out/"windows.csv", index=False)
    diagnostics = []
    for client, g in metadata[metadata.partition=="train"].groupby("client"):
        indices = g.index.to_numpy()
        m = np.asarray(arrays["target"])[indices]
        c = np.asarray(arrays["context"])[indices]
        diagnostics.append(dict(client=int(client), windows=len(g), receiver_sessions=g.session.nunique(),
            scenarios=g.scenario.nunique(), mean_speed_mps=float(m[:, 2].mean()*SCALES["movement"][2]),
            mean_neighbor_count=float(c[:, 0].mean()*SCALES["context"][0])))
    pd.DataFrame(diagnostics).to_csv(out/"client_diagnostics.csv", index=False)
    dump_json(out/"preparation.json", dict(origin=audit["origin"], config=cfg, partitions=counts,
        features=NAMES, fixed_scales=SCALES, road_map=bool(road.segments),
        raw_messages=len(frame), scored_windows=len(meta),
        window_coverage=len(meta)/len(frame), feature_processing_seconds=elapsed,
        average_feature_ms=elapsed/len(frame)*1000,
        client_train_counts=metadata[metadata.partition=="train"].client.value_counts().sort_index().to_dict(),
        note="Timing duplicates are preserved. Ground truth is excluded from features. Warm-up messages are not scored."))
    print(f"Prepared {len(meta)} windows: {counts}", flush=True)
    return out


def load_prepared(path):
    path = Path(path)
    with np.load(path/"windows.npz", allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    meta = pd.read_csv(path/"windows.csv", dtype={"pseudonym": str, "receiver": str, "message_id": str})
    info = read_json(path/"preparation.json")
    if not all(len(v) == len(meta) and np.isfinite(v).all() for v in data.values()):
        raise ValueError("Prepared data invalid or metadata length mismatch")
    return data, meta, info
