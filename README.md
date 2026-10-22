# FedContextTrust-V2X

A modular, runnable research prototype for benign-only, three-stream V2X anomaly detection.

**Start here:** use the synthetic demo to check your installation. Then run the separate VeReMi workflow. Synthetic numbers are not research results. Included real-data pilot results cover only one scenario, one attack family and short training.

**New: real RSU/server weight exchange.** Run `python scripts/network_smoke_test.py --out runs/network_smoke` for a tested separate-process HTTP demo. See `docs/NETWORK_FEDERATION.md` for manual terminals, multiple machines, TLS, and real-data commands. The existing single-process workflow remains available.

## What is implemented

| Component | Actual implementation |
|---|---|
| Dataset reader | Extracted VeReMi NextGen JSON arrays, including string-valued numbers/positions; see `docs/INGEST_FORMAT.md` |
| Label handling | Explicit baseline assumption for official attack-free archives; missing labels in attack sources cause an error |
| Movement | GRU(6,32), Dense(32,16), ReLU, Dense(16,6); predicts current movement from 10 previous observations |
| Communication timing | GRU autoencoder, 7 inputs, hidden size 24; reconstructs the last 10 timing observations |
| Neighbour / optional road context | MLP autoencoder: 10 → 16 → 8 → 4 → 8 → 16 → 10 |
| Training | Adam + MSE, benign-only fitting; simulation uses benign model selection, network mode uses fixed rounds |
| Federated learning | Existing single-process FedAvg plus separate RSU/server HTTP(S) processes with sample-weighted averaging |
| Calibration | Separate benign reference and calibration partitions, empirical-rank normalization, equal-weight fusion |
| Decisions | Trusted / Uncertain / Suspicious using two validation quantiles |
| Comparisons | Centralized vs federated, Isolation Forest, one-class SVM, all 7 nonempty stream combinations |
| Evaluation | Precision, recall, F1, FPR, ROC-AUC, AP, trapezoidal PR-AUC, uncertainty, per-attack results, coverage, episode delays |
| Efficiency | CPU inference median/p95, parameter count, checkpoint size, FL payload bytes |
| Reuse | Saved checkpoints, calibration, repeatable CLI, local streaming inference API and automated tests |

This is **not** a production V2X security system. It does not implement cryptographic secure aggregation, differential privacy, physical RSU integration, radio simulation, actual OBU deployment, certificate revocation or vehicle control. Federated learning alone is not a formal privacy guarantee. Road context requires a correctly aligned map; the default runs are neighbour-context-only.

## 1. Windows installation (PowerShell)

Use Python 3.11 or 3.12. Open the extracted `FedContextTrust_V2X` folder in VS Code. Run these commands in its terminal:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
```

Using the full Python path avoids PowerShell activation-policy problems. If you already have compatible PyTorch, the separate torch installation is unnecessary. A CUDA GPU is not required; the prototype deliberately uses CPU for reproducible latency measurements.

Linux/macOS equivalent:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pytest -q
```

`pyproject.toml` gives supported dependency ranges. `VERIFIED_ENVIRONMENT.txt` records the versions used for the delivered verification. It is a record, not a cross-platform lock file.

## 2. Run the complete demo

```powershell
.\.venv\Scripts\python.exe -m fedcontext demo --out runs/my_demo --config configs/demo.yaml --seeds 7 19 43
```

The command generates toy raw data, prepares causal features, trains both neural methods for three seeds, runs both classical baselines, evaluates and writes reports. No download is needed.

Open:

- `runs/my_demo/results/REPORT.md` — explanation of the generated results.
- `runs/my_demo/results/summary.csv` — means, standard deviations and number of seeds.
- `runs/my_demo/results/results_table.tex` — a simple LaTeX results table.
- `runs/my_demo/results/federated_seed7/` — detailed neural results and figures.

Every output directory must be new. For a second run choose `runs/my_demo_v2`; the program will not overwrite earlier experiments.

`examples/verified_demo/` contains measured outputs from the supplied code. Regenerate raw/prepared data with the command above. No external dataset is bundled.

## 3. Run your real VeReMi NextGen data

Download official archives from https://zenodo.org/records/19665762 or follow https://github.com/VeReMi-dataset/VeReMi-NextGen . Start with the small `InTAS_highway_2.zip` baseline and a matching attack ZIP.

Fully extract the baseline into `data/raw/benign/` and the attacks into `data/raw/attacks/`, including any inner ZIPs. Keep the official split/scenario folders and receiver filenames. **Keep all ZIPs outside these source directories.** The simplified reader requires extracted JSON arrays; see `docs/INGEST_FORMAT.md` for the format and migration steps.

The manifest is `configs/veremi_manifest.example.json`. Paths inside it are relative to the manifest, not the current terminal directory. Only the baseline source has `assume_benign: true`: official baseline records can omit `attacker`. Attack files must contain a valid 0/1 label. An attack archive still contains benign messages; folder membership is not a per-message attack label.

```powershell
.\.venv\Scripts\python.exe -m fedcontext prepare --manifest configs/veremi_manifest.example.json --config configs/research.yaml --out data/processed_real
.\.venv\Scripts\python.exe -m fedcontext run --prepared data/processed_real --config configs/research.yaml --out runs/real_experiment --seeds 7 19 43
```

For a quick installation/data check use `configs/demo.yaml` in **both** commands and a different output path. Call those outputs a short pilot, not the final experiment.

The primary protocol uses baseline Train for training; baseline Validation is split by whole receiver files into model selection, score-reference and threshold-calibration sets. Baseline Test and attack Test are evaluated. Attack Train/Validation are intentionally excluded. Official split membership is preserved, not randomized at message/window level.

Read the ingestion audit before training. Unknown labels or schema errors stop preparation with examples. Do not fix errors by replacing unknown labels with 0 or 1. The original Person 1 label failure is addressed by the explicit baseline-source policy, not by weakening validation.

RAM note: raw files are read receiver-by-receiver, but the prepared table and window arrays are currently held in memory. Start with one low-density scenario on a laptop. Full high-density collections may need substantial RAM or a future disk-backed pipeline. No automatic sample reduction occurs.

## 4. Optional road-map context

Use your own correctly projected map, following `docs/MAP_AND_FEATURES.md`. Raw `distance_to_road_edge` is deliberately not consumed; road consistency is recomputed from reported coordinates and an external map.

```powershell
.\.venv\Scripts\python.exe -m fedcontext prepare --manifest configs/veremi_manifest.example.json --config configs/research.yaml --road-map data/roads.json --out data/processed_with_map
```

Do not use unrelated coordinates or claim map-aware results from the default neighbour-only run.

## 5. Reuse a saved model

```powershell
.\.venv\Scripts\python.exe -m fedcontext replay --prepared data/processed_real --run runs/real_experiment/federated_seed7 --out runs/reloaded_predictions.csv
```

For one receiver session, `predict-stream` accepts canonical observable JSONL records. See `docs/INFERENCE.md`. Never load model files or joblib baselines from untrusted sources.

To evaluate an already-trained neural model on new held-out data without refitting weights or thresholds:

```powershell
.\.venv\Scripts\python.exe -m fedcontext evaluate --prepared data/processed_real --run runs/real_experiment/federated_seed7 --out runs/frozen_model_evaluation
```

## Folder guide

| Folder / module | Purpose |
|---|---|
| `configs/` | Demo/research settings and dataset manifest |
| `src/fedcontext/ingest.py` | Extracted JSON data, labels, provenance and split checks |
| `features.py`, `prepare.py` | Causal streams, optional map matching and windows |
| `models.py` | The three neural networks |
| `training.py` | Centralized training, local client training and FedAvg |
| `calibration.py` | Error calibration, score fusion and decision thresholds |
| `baselines.py` | Isolation Forest and one-class SVM |
| `metrics.py`, `experiment.py`, `reporting.py` | Experiments, evaluation, CSV/plots/LaTeX |
| `inference.py` | Reloading and receiver-local streaming inference |
| `demo.py` | Clearly labelled synthetic fixtures |
| `tests/` | Automated correctness and leakage checks |
| `docs/` | Methodology, interpretation, feature schema, viva and limitations |
| `examples/` | Actual verified demo and real-pilot outputs |

## What you can honestly tell your teacher

“We have implemented and tested a modular research prototype. It includes the three anomaly models, simulated RSU-level federated training and an evaluation pipeline. We have completed a synthetic end-to-end check and a small real-data pilot. Full multi-scenario evaluation, map validation, privacy mechanisms and actual vehicle deployment are still pending.”

See `docs/EVALUATION.md` before writing your results section and `docs/VERIFICATION.md` for exactly what was tested.
