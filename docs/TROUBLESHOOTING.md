# Troubleshooting

## “No module named fedcontext”

Run `python -m pip install -e ".[dev]"` using the same virtual-environment interpreter that runs the program. The new package is `fedcontext`, not the old `fedcontext_person1`.

## “No extracted JSON files under ...”

Check paths in `configs/veremi_manifest.example.json`. They are relative to that file. Run from the project root. No extra `data/data/raw` folder is needed.

## “Missing attacker label”

Read `ingestion_audit.json`. Official attack-free baseline archives can omit labels; put only verified clean baseline archives in the manifest's baseline source with `assume_benign: true`. Never enable that for attack data. Do not simply label every missing value benign.

## ZIPs still appear inside Train/Validation/Test

The simplified importer requires extracted JSON arrays. Fully extract inner ZIPs in a staging directory, then place only the extracted data tree under the raw source. Keep original archives outside the raw source. See `INGEST_FORMAT.md`.

## “Cannot identify exactly one split”

Keep the official Train, Validation and Test folders. For a manually managed source, set `split` explicitly to `train`, `validation` or `test` in that manifest source. Never combine multiple splits under one overriding value.

## “Unknown attack family”

Use an official attack name in the path or provide a descriptive `variant` value in the source manifest. That value labels evaluation groups only. It never replaces message labels.

## Too few selection / score-reference / calibration windows

The code splits whole validation receiver files into three disjoint roles. Supply more files or scenarios. For a clearly marked smoke test only, lower `calibration_min_windows` or history length and re-prepare into a new directory. Do not use test data to make up the difference.

## Only one active FL client

Inspect `client_diagnostics.csv` and `preparation.json`. A small geographic subset may occupy a single grid cell. Adjust region width or use receiver-hash partitioning for a documented simulation, then prepare again. Do not claim five participating RSUs when only one bucket contains windows.

## Output directory already exists

Choose a new path, such as `runs/real_experiment_v2`. The program deliberately does not overwrite existing work. Keep prior reports for comparison.

## Low F1 or high false-positive rate

First check labels, units, coverage, scenario shifts and the score distributions. Compare stream ablations and benign-only scenario FPR. Longer training or a different architecture might help, but thresholds must be chosen using validation, not tuned to the test set. A weak result is not an installation failure.

## Program runs out of memory

Use one low-density scenario first. The current prepared arrays are in memory; the full high-density collection is not a guaranteed laptop-sized workload. Do not silently sample test data. If using an explicit subset for a pilot, name its exact source files and scope.

## Road features remain zero

Without an external map this is intentional: `has_road_map=0`. This is neighbour-only context mode. Adding a file with an unrelated coordinate system will produce misleading results; verify map alignment first.
