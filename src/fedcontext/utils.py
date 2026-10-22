from pathlib import Path
import hashlib
import json
import random
import gzip
import numpy as np
import torch


def dump_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False, default=str), encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def seed_everything(seed, threads=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def stable_hash(text):
    return int(hashlib.sha256(str(text).encode()).hexdigest()[:16], 16)


def fresh_dir(path):
    p = Path(path)
    if p.exists() and any(p.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {p}. Choose a NEW output directory.")
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_csv(frame, path):
    """Close/flush the complete stream before publishing a CSV artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+".partial")
    if path.suffix == ".gz":
        with gzip.open(temporary, "wt", encoding="utf-8", newline="") as sink:
            frame.to_csv(sink, index=False)
    else:
        with temporary.open("w", encoding="utf-8", newline="") as sink:
            frame.to_csv(sink, index=False)
    temporary.replace(path)
