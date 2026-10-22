# FedContextTrust: paper figures and interpretation

This package generates figures from your completed full-network experiment. It does not train the models or change any saved results. Six architecture diagrams are included now. Measured result graphs are created when you run the script on your actual output files.

## 1. Put this folder in your project

Extract this ZIP. Copy the `Paper_Figures` folder into the full `FedContextTrust_V2X` project folder, beside `pyproject.toml`, `src`, `data` and `runs`.

Open PowerShell in that project root. Use the same working virtual environment as your completed run.

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\Paper_Figures\requirements.txt
```

This installs plotting tools. You do not need to retrain or run `finalize` again if your evaluation outputs already exist.

## 2. Locate your outputs

```powershell
Get-ChildItem .\runs -Recurse -Filter predictions.csv.gz | Select-Object FullName
Get-ChildItem .\runs -Recurse -Filter client_rounds.csv | Select-Object FullName
Get-ChildItem .\runs -Recurse -Filter calibration.json | Select-Object FullName
```

The following command uses the output names from your federated README. Change these paths if your run names differ. The wildcard must select only the RSUs from this one experiment. Never combine different runs or seeds.

```powershell
.\.venv\Scripts\python.exe .\Paper_Figures\generate_figures.py `
  --evaluation .\runs\federated_final_v1\evaluation `
  --calibration .\runs\federated_final_v1\calibration.json `
  --clients ".\runs\rsu_*_v1\client_rounds.csv" `
  --cloud .\runs\cloud_v1\communication.csv `
  --out .\runs\paper_figures_v1
```

Each PowerShell backtick must be the last character on its line, without spaces after it. Alternatively paste the command as one line without backticks.

`--evaluation` selects the saved held-out test results. `--calibration` supplies the exact fixed thresholds used by that detector. `--clients` selects the local training logs. `--cloud` supplies round timing. `--out` creates a new figure folder. Existing output folders are protected; use `paper_figures_v2` for a repeat.

The program needs no raw dataset and no model checkpoint. It reads the results already computed by your implementation. The first run may take a few minutes because every figure is exported in three formats.

## 3. Open the figures and explanations

```powershell
Invoke-Item .\runs\paper_figures_v1\report.html
Invoke-Item .\runs\paper_figures_v1
```

The HTML report is a local file. Open it in Chrome, Edge or Firefox. It contains the generated result plots, captions, interpretation and notices about any unavailable data. It can be printed to PDF from your browser if desired.

Each figure has:

- `.pdf`: vector output, suitable for a paper or LaTeX.
- `.svg`: editable vector output.
- `.png`: 300 dpi output, suitable for Word and slides.

Also produced: `INTERPRETATION.md`, `figure_manifest.json` with input hashes, `recomputed_policy_metrics.csv`, `threshold_sweep_descriptive_only.csv`, and other supporting tables where applicable. Keep the input files and manifest with your experiment record. The manifest contains local paths; review these before public release.

## 4. Which figures should go in the paper?

Use a small main set and put additional diagnostics in supplementary material. More figures do not automatically make a stronger paper.

| Main-paper figure | What it demonstrates |
|---|---|
| D2 + D3 or a combined architecture figure | The three models and the actual federated learning loop |
| T_local_losses or T_weighted_local_losses | How local benign-data fitting changes across rounds |
| R_ROC + R_precision_recall | How well the final anomaly scores separate benign and attack windows |
| R_confusion_strict + R_confusion_alert | Actual errors at the saved operating thresholds |
| R_stream_subsets | Whether adding evidence streams helps on this test set |
| R_attack_groups | Attack types with good detection and those still missed |
| F_payload_cumulative | Model payload cost of distributed training |

The PDF guide explains the methodology diagrams and a complete figure catalogue. The generated report gives the interpretation of your measured results. You must check those values before putting them into a paper.

## 5. Complete available result figure catalogue

Up to **23 result figures plus 6 architecture diagrams**, depending on available columns and classes:

| Figure filename | Read it this way |
|---|---|
| R_ROC | Higher recall at lower false-positive rates is better. ROC-AUC measures ranking across thresholds. |
| R_precision_recall | High precision at useful recall is desirable. AP is average precision; the horizontal baseline is attack prevalence. |
| R_error_tradeoff | Smaller false-negative and false-positive rates are better. A single threshold usually trades one for the other. |
| R_confusion_strict | Only Suspicious is positive. Count correct detections, missed attacks and false alarms. |
| R_confusion_alert | Both Uncertain and Suspicious are positive. Usually catches more attacks at a higher false-alarm rate. |
| R_policy_metrics | Accuracy, precision, recall, F1 and FPR for both policies. FPR is the one where lower is better. |
| R_threshold_sensitivity | Describes how metrics change with score threshold. Never use the test optimum to tune the published detector. |
| R_score_distribution | Better class separation supports detection; overlap explains errors. |
| R_score_ECDF | At t, attack ECDF is the fraction missed by score > t; 1 - benign ECDF is FPR. |
| R_three_decisions | Shows the proportions assigned Trusted, Uncertain and Suspicious within each actual class. |
| R_stream_mse | Shows error distributions separately for movement, timing and context. Different streams have different raw scales. |
| R_stream_rank | Shows comparable empirical-rank distributions after reference calibration. |
| R_stream_correlation | Shows whether streams tend to rank the same windows similarly. Correlation alone is not an ablation. |
| R_attack_groups | Strict and alert recall for each saved attack group; denominators shown. |
| R_sender_timeline | A clearly identified selection rule chooses one sender/session for a temporal case study. |
| R_stream_subsets | Compares the saved single-stream, paired-stream and full-stream evaluations; no retraining is implied. |
| T_local_losses | Local loss for every supplied RSU; not test accuracy or global validation loss. |
| T_weighted_local_losses | Window-count-weighted summary of those local losses. It does not evaluate the averaged global model. |
| T_client_sizes | Shows sample-count weighting; it does not prove non-IID feature distributions. |
| F_payload_per_round | Recorded upload and download model-body MiB per round. |
| F_payload_cumulative | Accumulated model-body MiB over training rounds. |
| F_client_duration | Local training plus upload duration; shared-laptop contention affects this. |
| F_round_duration | Synchronous round duration observed by the cloud coordinator. |

## 6. Exact required inputs and checks

Required file: `evaluation/predictions.csv.gz` (or uncompressed `predictions.csv`). Required columns: `label`, `score`, `decision`. Labels must be 0=benign or 1=attack; decisions must be Trusted, Uncertain or Suspicious. Scores must be finite in [0,1]. Only scored held-out test windows belong here.

Optional stream columns: `movement_mse`, `timing_mse`, `context_mse`, `movement_rank`, `timing_rank`, `context_rank`. The timeline needs `session`, `receiver`, `pseudonym`, `rx_time`. Attack grouping uses `variant`, falling back to `scenario`.

`metrics.csv` needs `subset`, `policy`, `f1`, `recall`, `fpr` for the stream comparison. If your release uses different column names, the report will explain the missing columns; share the header rather than guessing a mapping.

Training logs need `round`, `samples`, `movement_loss`, `timing_loss`, `context_loss`. Optional payload fields are `download_body_bytes`, `upload_body_bytes`; duration is `local_train_and_upload_seconds`.

Cloud duration uses `round` and `round_seconds` in `communication.csv`.

Saved calibration is expected as `thresholds["movement+timing+context"] = [low, high]`. The script recomputes decisions using these thresholds and stops if they disagree with the CSV. This catches accidentally mixed experiment files. It never changes thresholds to make the figures look better.

## 7. Theory in simple words

The movement model predicts the next movement. The timing and context autoencoders reconstruct their inputs. Each compares its output with its target and produces mean squared error (MSE).

The cloud combines matching parameters from RSUs using FedAvg: the contribution of an RSU is its training-window count divided by the total count. Averaging happens between copies of the same model, not between movement and timing architectures.

After training, normal reference windows establish typical error distributions. Separate normal calibration windows set the lower and upper score thresholds. A new window's three reference ranks are averaged. That fused score is compared with the fixed thresholds. A rank is not an attack probability.

For evaluation, label 1 means attack. TP is a detected attack; FN is a missed attack; FP is benign data flagged positive; TN is benign data left negative.

- Precision = TP / (TP + FP): among positive alerts, how many are attacks?
- Recall = TP / (TP + FN): among attacks, how many are detected?
- F1 = 2TP / (2TP + FP + FN): balances precision and recall.
- FPR = FP / (FP + TN): how many benign windows are flagged?
- Accuracy = (TP + TN) / all evaluated windows: overall correctness, sensitive to class balance.

Undefined metrics remain missing rather than being invented. There is no third actual class called Uncertain.

## 8. Results that need extra experiments

Do not create accuracy-vs-round curves by converting MSE into accuracy. The network logs contain no held-out global evaluation at each round, and only the latest checkpoint is retained. A future run must save/evaluate each global checkpoint on a fixed validation set to support a validation-accuracy curve. Keep the test set for the final locked evaluation.

Likewise, actual-versus-predicted movement traces require per-feature model outputs and their targets. The existing error-only prediction export is insufficient. End-to-end OBU latency requires hardware and feature-extraction timing; a training-duration chart cannot establish it.

For stronger claims, add repeated independent seeds, matched local-only and centralized baselines, retrained stream ablations, held-out attack families, and confidence intervals clustered by independent sessions/runs. Overlapping windows are not independent trials. Do not use naive per-window resampling to claim narrow confidence intervals.

The network version does not use road-map data. It does not implement secure aggregation, differential privacy or automatic online retraining. The current neighbour context stream remains active. Architectural diagrams describe this implementation and intended OBU placement, not a physical OBU deployment.

## 9. Send the actual results for interpretation

To let the assistant inspect and generate the final measured plots, upload a ZIP containing these files from the SAME experiment:

- `runs/federated_final_v1/evaluation/` (including predictions and metrics).
- `runs/federated_final_v1/calibration.json`.
- Each `runs/rsu_<id>_v1/client_rounds.csv`.
- `runs/cloud_v1/communication.csv` and `training.json`.

No raw dataset, model checkpoint, `client.json`, bearer tokens or `server/federation.json` is needed. Optional `coverage.json`, `latency.json`, `episode_delays.csv` and episode summaries can support further carefully defined figures after their contents are inspected.

For the default run names, a collector is included. From the project root run:

```powershell
.\Paper_Figures\Collect-Results.ps1
```

It creates `results_for_figures.zip` using an explicit allowlist of result files. Upload that ZIP. If your paths differ, specify `-FinalRun`, `-ClientPattern` and `-CloudRun`, for example:

```powershell
.\Paper_Figures\Collect-Results.ps1 -FinalRun runs/federated_final_v2 -ClientPattern 'runs/rsu_*_v2/client_rounds.csv' -CloudRun runs/cloud_v2 -Out results_v2.zip
```

If your computer does not allow PowerShell scripts, create the ZIP manually from the listed result files. No policy change is needed.

Until those measured outputs are available, quantitative conclusions and paper result values stay unfilled. The diagrams included here contain architecture and method facts, not fabricated experiment results.

## 10. Sources and implementation basis

- Supplied `README_FEDERATED.md`: network flow, three stream tasks, training logs, saved outputs, and implementation limits.
- McMahan et al., Communication-Efficient Learning of Deep Networks from Decentralized Data, AISTATS 2017: https://proceedings.mlr.press/v54/mcmahan17a.html
- scikit-learn, Model evaluation: https://scikit-learn.org/stable/modules/model_evaluation.html

These references support the method and metric definitions. They do not provide results for this project. Existing five related-work papers are not experimental baselines unless you actually reproduce/evaluate them under a matched protocol.
