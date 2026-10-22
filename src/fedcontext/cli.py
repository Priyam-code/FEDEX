import argparse
from pathlib import Path
import sys
import json
from .config import load_config
from .utils import fresh_dir


def main():
    p = argparse.ArgumentParser(prog="fedcontext", description="FedContextTrust-V2X research prototype")
    sub = p.add_subparsers(dest="command", required=True)
    d = sub.add_parser("demo", help="Generate fixtures, prepare, train, evaluate and export reports")
    d.add_argument("--out", default="runs/demo")
    d.add_argument("--config", default="configs/demo.yaml")
    d.add_argument("--seeds", nargs="+", type=int, default=[7, 19, 43])
    d.add_argument("--modes", nargs="+", choices=["centralized", "federated"], default=["centralized", "federated"])
    g = sub.add_parser("generate-demo")
    g.add_argument("--out", required=True)
    a = sub.add_parser("prepare", help="Read JSON/nested ZIP and build causal windows")
    a.add_argument("--manifest", required=True)
    a.add_argument("--out", default="data/processed")
    a.add_argument("--config", default="configs/research.yaml")
    a.add_argument("--road-map")
    r = sub.add_parser("run", help="Train/calibrate/evaluate; all output folders must be new")
    r.add_argument("--prepared", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--config", default="configs/research.yaml")
    r.add_argument("--modes", nargs="+", choices=["centralized", "federated"], default=["centralized", "federated"])
    r.add_argument("--seeds", nargs="+", type=int, default=[7, 19, 43])
    r.add_argument("--no-baselines", action="store_true")
    s = sub.add_parser("summarize")
    s.add_argument("--runs", required=True)
    replay = sub.add_parser("replay", help="Reload saved model and reproduce test predictions")
    replay.add_argument("--prepared", required=True)
    replay.add_argument("--run", required=True)
    replay.add_argument("--out", required=True)
    ev = sub.add_parser("evaluate", help="Evaluate an existing checkpoint with frozen calibration; no retraining")
    ev.add_argument("--prepared", required=True)
    ev.add_argument("--run", required=True)
    ev.add_argument("--out", required=True)
    stream = sub.add_parser("predict-stream", help="Canonical JSONL, one receiver/session, receive-time order")
    stream.add_argument("--run", required=True)
    stream.add_argument("--input", required=True)
    stream.add_argument("--out", required=True)
    stream.add_argument("--road-map")
    args = p.parse_args()
    try:
        if args.command == "generate-demo":
            from .demo import make_demo
            print(make_demo(args.out))
        elif args.command == "demo":
            from .demo import make_demo
            from .prepare import prepare
            from .experiment import run
            cfg = load_config(args.config)
            root = fresh_dir(args.out)
            manifest = make_demo(root/"raw")
            prepare(manifest, root/"processed", cfg)
            run(root/"processed", root/"results", cfg, args.modes, args.seeds)
            print(f"DEMO COMPLETE — synthetic only. Open {root/'results'/'REPORT.md'}")
        elif args.command == "prepare":
            from .prepare import prepare
            prepare(args.manifest, args.out, load_config(args.config), args.road_map)
        elif args.command == "run":
            from .experiment import run
            run(args.prepared, args.out, load_config(args.config), args.modes, args.seeds, not args.no_baselines)
            print(f"Complete: {Path(args.out)/'REPORT.md'}")
        elif args.command == "summarize":
            from .reporting import summarize
            summarize(args.runs)
        elif args.command == "replay":
            from .inference import replay
            replay(args.prepared, args.run, args.out)
        elif args.command == "evaluate":
            from .experiment import evaluate_saved
            evaluate_saved(args.prepared, args.run, args.out)
        elif args.command == "predict-stream":
            from .inference import StreamingDetector
            detector = StreamingDetector(args.run, args.road_map)
            destination = Path(args.out)
            if destination.exists():
                raise FileExistsError(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with Path(args.input).open(encoding="utf-8") as source, destination.open("x", encoding="utf-8") as sink:
                for line in source:
                    if line.strip():
                        sink.write(json.dumps(detector.update(json.loads(line)), allow_nan=False)+"\n")
    except (ValueError, FileNotFoundError, FileExistsError, KeyError) as exc:
        p.exit(2, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
