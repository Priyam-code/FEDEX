"""Strict ingestion of extracted VeReMi-style JSON arrays. See docs/INGEST_FORMAT.md."""
import hashlib
import io
import re
from pathlib import Path
import numpy as np
import pandas as pd
import ijson
from .utils import read_json, dump_json, save_csv

ATTACKS = ["constantPositionOffset", "randomPositionOffset", "positionMirroring",
           "constantSpeedOffset", "randomSpeedOffset", "zeroSpeedReport", "suddenConstantSpeed",
           "reversedHeading", "feignedBraking", "accelerationMultiplication", "timeDelayAttack",
           "dosAttack", "trafficCongestionSybil", "suddenStop", "dataReplay"]


def vector(value):
    if isinstance(value, str):
        value = value.strip("[]()").split(",")
    a = np.asarray(value, dtype=float)
    if a.ndim != 1 or len(a) not in (2, 3) or not np.isfinite(a).all():
        raise ValueError("Expected a finite x,y[,z] position")
    return a[:2]


def label(value, assume_benign=False):
    if value is None:
        if assume_benign:
            return 0
        raise ValueError("Missing attacker label. Only a manifest-declared clean baseline may assume 0.")
    if isinstance(value, bool):
        return int(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Unsupported attacker label: {value!r}; expected 0 or 1") from None
    if number not in (0, 1):
        raise ValueError(f"Unsupported attacker label: {value!r}")
    return int(number)


def records(body):
    """Yield messages from one UTF-8 JSON array; no format guessing."""
    stripped = body.removeprefix(b"\xef\xbb\xbf").lstrip()
    if not stripped.startswith(b"["):
        raise ValueError("Expected a JSON array: [{message}, ...]. JSONL and wrappers are not supported.")
    try:
        yield from ijson.items(io.BytesIO(stripped), "item", use_float=True)
    except ijson.JSONError as exc:
        raise ValueError(f"Invalid JSON array: {exc}") from exc


def raw_files(base):
    """Require extracted JSON files. Reject leftover archives instead of ignoring them."""
    paths = [base] if base.is_file() else sorted(p for p in base.rglob("*") if p.is_file())
    unsupported = next((p for p in paths if p.suffix.lower() in (".zip", ".jsonl", ".gz", ".bz2", ".xz", ".7z", ".tar")), None)
    if unsupported:
        raise ValueError(f"Extract archives outside the raw source and use JSON arrays only: {unsupported}")
    paths = [p for p in paths if p.suffix.lower() == ".json"]
    if not paths:
        raise FileNotFoundError(f"No extracted JSON files under {base}")
    return paths


def parse_message(obj, assume_benign=False):
    """Validate one message and return only the canonical model/label fields."""
    if not isinstance(obj, dict):
        raise ValueError("Each array entry must be a message object")
    sender, receiver = obj["sender"], obj["receiver"]
    if not isinstance(sender, dict) or not isinstance(receiver, dict):
        raise ValueError("sender and receiver must be objects")
    alias, message_id = obj.get("sender_alias"), obj.get("messageID")
    for key, value in (("sender_alias", alias), ("messageID", message_id)):
        if type(value) not in (str, int) or not str(value).strip():
            raise ValueError(f"{key} must be a non-empty string or integer")
    x, y = vector(sender["pos"])
    rx_x, rx_y = vector(receiver["pos"])
    numeric = dict(rx_time=float(obj["rcvTime"]), tx_time=float(obj["sendTime"]),
        x=x, y=y, rx_x=rx_x, rx_y=rx_y, speed=float(sender["spd"]),
        acceleration=float(sender["acl"]), heading=float(sender["hed"]))
    if not np.isfinite(list(numeric.values())).all() or numeric["rx_time"] < 0 or numeric["tx_time"] < 0:
        raise ValueError("Numeric values must be finite and timestamps non-negative")
    numeric["heading"] %= 360
    return dict(pseudonym=str(alias), message_id=str(message_id),
                **numeric, label=label(obj.get("attacker"), assume_benign))


def infer_split(name):
    tokens = re.split(r"[/\\!]+", name.lower())
    found = {dict(train="train", validation="validation", val="validation", test="test")[t]
             for t in tokens if t in ("train", "validation", "val", "test")}
    if len(found) != 1:
        raise ValueError(f"Cannot identify exactly one Train/Validation/Test split in {name}")
    return found.pop()


def ingest(manifest_path, output):
    manifest_path = Path(manifest_path).resolve()
    manifest = read_json(manifest_path)
    if manifest.get("origin") not in ("synthetic_demo", "veremi_nextgen", "custom"):
        raise ValueError("Manifest origin must be synthetic_demo, veremi_nextgen, or custom")
    unit = manifest.get("timestamp_unit")
    if unit not in ("s", "ms", "us", "ns"):
        raise ValueError("Set timestamp_unit explicitly: s, ms, us or ns")
    factor = dict(s=1, ms=1e-3, us=1e-6, ns=1e-9)[unit]
    rows, files, skipped, issues = [], [], [], []
    hashes = {}
    implicit = 0
    invalid_count = 0
    for source in manifest["sources"]:
        role = source["role"]
        if role not in ("baseline", "attack"):
            raise ValueError("Source role must be baseline or attack")
        assume = source.get("assume_benign", False)
        if not isinstance(assume, bool):
            raise ValueError("assume_benign must be a JSON boolean: true or false")
        if assume and role != "baseline":
            raise ValueError("assume_benign is forbidden for attack sources")
        base = (manifest_path.parent / source["path"]).resolve()
        for path in raw_files(base):
            name = str(path)
            logical_name = path.name if base.is_file() else base.name+"/"+path.relative_to(base).as_posix()
            if "ground_truth" in name.lower() or "groundtruth" in name.lower():
                skipped.append(dict(file=name, reason="ground truth is not model input"))
                continue
            split = source.get("split") or infer_split(logical_name)
            if split not in ("train", "validation", "test"):
                raise ValueError(f"Invalid split {split}")
            if role == "attack" and split != "test":
                skipped.append(dict(file=name, reason="attack train/validation excluded by protocol"))
                continue
            if path.stat().st_size > 256_000_000:
                raise ValueError(f"JSON file exceeds 256 MB: {path}")
            body = path.read_bytes()
            digest = hashlib.sha256(body).hexdigest()
            if digest in hashes:
                if hashes[digest] != split:
                    raise ValueError(f"Identical file content across splits: {name}")
                skipped.append(dict(file=name, reason="exact duplicate file content"))
                continue
            hashes[digest] = split
            scenario_match = re.search(r"(?:InTAS_)?(?:highway|urban)_[27]", logical_name, re.I)
            scenario = source.get("scenario") or (scenario_match.group(0).replace("InTAS_", "") if scenario_match else "custom")
            variant = source.get("variant") or ("benign" if role == "baseline" else next((a for a in ATTACKS if a.lower() in logical_name.lower()), None))
            if variant is None:
                raise ValueError(f"Unknown attack family in {name}; set source.variant in manifest")
            receiver = path.stem
            session_key = f"{scenario}:{variant}:{split}:{receiver}:{digest}"
            session = hashlib.sha256(session_key.encode()).hexdigest()[:16]
            count = 0
            for idx, obj in enumerate(records(body)):
                try:
                    fields = parse_message(obj, assume)
                    implicit += int(obj.get("attacker") is None)
                    if role == "baseline" and fields["label"] != 0:
                        raise ValueError("Attack-labelled record in baseline source")
                    fields["rx_time"] *= factor
                    fields["tx_time"] *= factor
                    rows.append(dict(record_id=f"{session}:{idx}", session=session, receiver=receiver,
                        split=split, scenario=scenario, variant=variant, role=role,
                        **fields, rsu=str(source.get("rsu", "")), source_file=name))
                    count += 1
                except (ValueError, KeyError, TypeError, OverflowError) as exc:
                    invalid_count += 1
                    if len(issues) < 100:
                        issues.append(dict(file=name, index=idx, error=str(exc)))
            files.append(dict(file=name, sha256=digest, split=split, role=role, records=count))
    audit = dict(origin=manifest["origin"], files=files, skipped_files=skipped,
                 invalid_examples=issues, invalid_records=invalid_count, manifest_assumed_benign_records=implicit,
                 parsed_records=len(rows), timestamp_unit=unit)
    dump_json(Path(output)/"ingestion_audit.json", audit)
    if issues:
        raise ValueError("Invalid records detected. See ingestion_audit.json. No silent label repair or row dropping.")
    if not rows:
        raise ValueError("No messages parsed")
    frame = pd.DataFrame(rows)
    # A replay across official splits must not be learned and evaluated twice.
    identity = ["scenario", "receiver", "pseudonym", "message_id", "rx_time", "tx_time"]
    if frame.groupby(identity, dropna=False)["split"].nunique().gt(1).any():
        raise ValueError("Message identity overlap across splits. Fix manifest/run provenance before training.")
    save_csv(frame, Path(output)/"messages.csv.gz")
    dump_json(Path(output)/"source_manifest.json", manifest)
    return frame, audit
