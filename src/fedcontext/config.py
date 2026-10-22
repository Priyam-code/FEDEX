from pathlib import Path
import yaml

DEFAULTS = dict(history=10, clients=5, partition="spatial", region_width_m=1000,
                max_gap_s=5, neighbor_ttl_s=1, neighbor_radius_m=300,
                epochs=40, rounds=20, local_epochs=2, batch_size=128,
                learning_rate=0.001, patience=5, min_delta=1e-6, threads=1,
                q_low=.95, q_high=.99, calibration_min_windows=200,
                baseline_max_samples=20000, latency_samples=500)


def load_config(path):
    cfg = DEFAULTS | (yaml.safe_load(Path(path).read_text()) or {})
    if set(cfg) - set(DEFAULTS):
        raise ValueError(f"Unknown config keys: {set(cfg)-set(DEFAULTS)}")
    for key in ["history", "clients", "epochs", "rounds", "local_epochs", "batch_size", "threads", "patience", "calibration_min_windows", "baseline_max_samples", "latency_samples"]:
        if not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ["region_width_m", "max_gap_s", "neighbor_ttl_s", "neighbor_radius_m", "learning_rate"]:
        if cfg[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if not 0 < cfg["q_low"] < cfg["q_high"] < 1:
        raise ValueError("Require 0 < q_low < q_high < 1")
    if cfg["partition"] not in ("spatial", "receiver_hash"):
        raise ValueError("partition must be spatial or receiver_hash")
    return cfg
