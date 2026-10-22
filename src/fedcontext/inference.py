"""Local CPU inference. No vehicle actuation or certificate revocation."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .models import ThreeStream
from .features import ReceiverFeatures, RoadMap
from .calibration import Calibrator
from .training import score
from .prepare import load_prepared
from .utils import read_json


def load_detector(run):
    run = Path(run)
    info = read_json(run/"run.json")
    torch.set_num_threads(info["config"]["threads"])
    model = ThreeStream()
    model.load_state_dict(torch.load(run/"model.pt", map_location="cpu", weights_only=True))
    model.eval()
    return model, Calibrator.from_dict(read_json(run/"calibration.json")), info


def replay(prepared, run, output):
    model, cal, info = load_detector(run)
    data, meta, prep = load_prepared(prepared)
    for k in ("history", "neighbor_ttl_s", "neighbor_radius_m", "max_gap_s"):
        if info["config"][k] != prep["config"][k]:
            raise ValueError(f"Feature configuration mismatch: {k}")
    if info["road_map"] != prep["road_map"]:
        raise ValueError("Road-map mode mismatch")
    ids = np.flatnonzero(meta.partition.eq("test"))
    errors = score(model, data, ids)
    scores, decisions, _ = cal.predict(errors)
    result = meta.iloc[ids].copy()
    result["score"], result["decision"] = scores, decisions
    p = Path(output)
    if p.exists():
        raise FileExistsError(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(p, index=False)


class StreamingDetector:
    """One detector per OBU/receiver session. Caller supplies canonical observable fields.

    Required: rx_time, tx_time (seconds), pseudonym, message_id, x,y,rx_x,rx_y (metres),
    speed (m/s), acceleration (m/s²), heading (degrees clockwise from north).
    Do not mix sessions/receivers in one instance. Labels are not needed.
    """
    def __init__(self, run, road_path=None):
        self.model, self.cal, self.info = load_detector(run)
        if bool(road_path) != self.info["road_map"]:
            raise ValueError("Use the same road-map mode as training")
        self.engine = ReceiverFeatures(self.info["config"], RoadMap(road_path))

    @torch.inference_mode()
    def update(self, observation):
        row = {k: observation[k] for k in ("rx_time", "tx_time", "pseudonym", "message_id", "x", "y", "rx_x", "rx_y", "speed", "acceleration", "heading")}
        numeric = [row[k] for k in row if k not in ("pseudonym", "message_id")]
        if not np.isfinite(np.asarray(numeric, dtype=float)).all():
            raise ValueError("Non-finite observation")
        f = self.engine.update(row)
        if f is None:
            return dict(decision="Uncertain", score=None, reason="insufficient_history")
        errors = self.model.errors(*(torch.from_numpy(f[k][None]) for k in ("movement", "target", "timing", "context"))).numpy()
        scores, decisions, ranked = self.cal.predict(errors)
        return dict(decision=str(decisions[0]), score=float(scores[0]), stream_mse=errors[0].tolist(),
                    stream_rank=ranked[0].tolist(), reason="model_score")
