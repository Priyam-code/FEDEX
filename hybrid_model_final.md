# Hybrid Model Final

## Purpose

This README runs the current hybrid detector in the existing **FedContextTrust-V2X** project. It covers calibration, evaluation, ROC curves, and prediction for an actual vehicle message.

The hybrid detector reuses the three trained neural models and adds revised scoring plus a speed–position consistency check. You do not need to repeat preprocessing or training for this update.

**Use `upgrade_scoring.py` and `profile="hybrid"` for this workflow. The original inference import and original evaluation commands still use the old scoring.**

## 1. Required files and folders

Place the supplied `upgrade_scoring.py` in the project root, beside `src`, `data`, and `runs`. This README does not contain that Python file; use the separately supplied script.

| Path | Purpose |
| --- | --- |
| `upgrade_scoring.py` | New calibration, evaluation, graphs, and hybrid prediction code |
| `.venv/Scripts/python.exe` | Existing Python environment |
| `src/fedcontext` | Original project modules |
| `data/veremi_processed` | Original prepared data, including benign `score_ref` and `calibration` partitions |
| `data/veremi_expanded_eval` | Prepared test data containing constant/random position and speed offsets |
| `runs/federated_final_v1` | Existing trained weights and original calibration |
| `runs/federated_scoring_v2` | Hybrid scoring artifact created in Step 3 |

The source model folder must contain `model.pt`, `run.json`, and `calibration.json`. Keep the existing environment with PyTorch, NumPy, pandas, scikit-learn, matplotlib, and the original project dependencies.

## 2. Open PowerShell in the project root

For the current project location:

```powershell
Set-Location "C:\Users\dcpri\Downloads\FedContextTrust_V2X_Implementation_v1_rudraupdate\FedContextTrust_V2X"
```

If you moved the project, use its new location. All following paths are relative to this folder. You do not need to activate the environment because each command calls its Python executable directly.

Optional check:

```powershell
Test-Path .\upgrade_scoring.py
Test-Path .\.venv\Scripts\python.exe
Test-Path .\runs\federated_final_v1\model.pt
Test-Path .\data\veremi_processed\windows.npz
Test-Path .\data\veremi_expanded_eval\windows.npz
```

All five should print `True`. Stop and correct any missing path before continuing.

> Run the commands in order. If a Python command reports an error, resolve it before running later steps. In multiline PowerShell commands, the backtick must be the last character on the line.

## 3. Create hybrid calibration once

```powershell
if (!(Test-Path .\runs\federated_scoring_v2\scoring_upgrade.json)) {
    .\.venv\Scripts\python.exe .\upgrade_scoring.py calibrate `
      --prepared data/veremi_processed `
      --run runs/federated_final_v1 `
      --out runs/federated_scoring_v2
}
```

If the hybrid artifact already exists, this block skips calibration.

### What happens

1. The script loads the existing trained models.
2. It calculates separate errors for the six movement outputs, timing, and context, plus endpoint speed–position consistency.
3. Benign `score_ref` windows establish each evidence channel's error distribution.
4. Separate benign `calibration` windows set the final decision thresholds.
5. It saves a new scoring artifact in `runs/federated_scoring_v2`.

No neural weights are trained or changed. No test attacks are used to fit these distributions or thresholds. The original model folder remains available for comparison.

The important new files are `scoring_upgrade.json` and `scoring_thresholds.csv`. The original weights and legacy calibration are copied into the new folder so the comparison can use identical weights.

**Recalibrate into a new folder if model weights or scoring logic change. Do not reuse this file with a different model.**

## 4. Evaluate hybrid and generate graphs

Paste the entire block into the same PowerShell window:

```powershell
$hybridResults = "runs/hybrid_results_$(Get-Date -Format 'yyyyMMdd_HHmmss')"

.\.venv\Scripts\python.exe .\upgrade_scoring.py evaluate `
  --prepared data/veremi_expanded_eval `
  --run runs/federated_scoring_v2 `
  --profile hybrid `
  --out "$hybridResults"
```

The timestamp gives each evaluation a fresh output folder. The script scores eligible test windows and generates predictions, metrics, and figures. It does not retrain or fit thresholds on test data.

### Command arguments

| Argument | Meaning |
| --- | --- |
| `evaluate` | Run the saved detector on test windows and generate results |
| `--prepared` | Prepared dataset to evaluate |
| `--run` | Folder containing trained weights and hybrid scoring configuration |
| `--profile hybrid` | Use hybrid as the primary prediction method |
| `--out` | New folder for this evaluation's outputs |

## 5. Display hybrid results

### Results for every attack family

```powershell
Import-Csv "$hybridResults/metrics_comparison.csv" |
  Where-Object {
    $_.method -eq "hybrid" -and $_.policy -eq "strict"
  } |
  Select-Object variant, roc_auc, attack_recall_pct, benign_fpr_pct, excluded_windows |
  Format-Table -AutoSize
```

### Overall hybrid results

```powershell
Import-Csv "$hybridResults/metrics_comparison.csv" |
  Where-Object {
    $_.method -eq "hybrid" -and
    $_.variant -eq "ALL" -and
    $_.policy -eq "strict"
  } |
  Select-Object roc_auc, average_precision, attack_recall_pct, benign_fpr_pct, precision_pct, f1 |
  Format-List
```

### Random-speed results only

```powershell
Import-Csv "$hybridResults/metrics_comparison.csv" |
  Where-Object {
    $_.method -eq "hybrid" -and
    $_.variant -eq "randomSpeedOffset" -and
    $_.policy -eq "strict"
  } |
  Format-List
```

### Reading the metrics

| Metric | Meaning |
| --- | --- |
| `roc_auc` | How well scores rank attacks above benign messages across thresholds; 0.5 is chance-level ranking |
| `average_precision` | Precision–recall summary; interpret alongside attack prevalence |
| `attack_recall_pct` | Percentage of attacks flagged under the selected policy |
| `benign_fpr_pct` | Percentage of benign messages incorrectly flagged |
| `precision_pct` | Percentage of flagged messages that are attacks |
| `f1` | Balance of precision and recall, on a 0–1 scale |
| `excluded_windows` | Windows omitted from the common valid-window comparison |

`strict` flags only **Suspicious** messages. `alert` flags both **Uncertain** and **Suspicious**. Replace `strict` with `alert` in the filters to view that policy.

The `ALL` row pools all test variants; it is not the arithmetic average of their AUCs. AUC is undefined for a benign-only group, so a blank/NaN value there is expected. AUC is not accuracy or the percentage of attacks detected.

## 6. Open ROC and precision–recall graphs

Open the figures folder:

```powershell
Invoke-Item "$hybridResults/figures"
```

Open random-speed ROC:

```powershell
Start-Process "$hybridResults/figures/randomSpeedOffset_ROC.png"
```

Open overall ROC:

```powershell
Start-Process "$hybridResults/figures/ALL_ROC.png"
```

Open random-speed precision–recall:

```powershell
Start-Process "$hybridResults/figures/randomSpeedOffset_PR.png"
```

PNG and PDF versions are saved. Each graph compares these methods:

| Curve | Scoring method |
| --- | --- |
| `original_fusion` | Original average of the three calibrated stream scores |
| `original_movement` | Original movement mean squared error |
| `featurewise` | Maximum calibrated evidence from six separate movement errors, timing error, and context error |
| `hybrid` | Featurewise scoring plus speed–position endpoint consistency |

For ROC, X is false positive rate and Y is attack recall. Curves closer to the upper-left corner generally indicate better ranking. For precision–recall, X is recall and Y is precision.

**Use the curve named `hybrid` for the current detector, while retaining the other curves as comparisons.**

## 7. Predict one actual vehicle message

### Message from the random-speed attack subset

```powershell
.\.venv\Scripts\python.exe .\upgrade_scoring.py predict-example `
  --prepared data/veremi_expanded_eval `
  --run runs/federated_scoring_v2 `
  --variant randomSpeedOffset `
  --profile hybrid
```

### Benign message

```powershell
.\.venv\Scripts\python.exe .\upgrade_scoring.py predict-example `
  --prepared data/veremi_expanded_eval `
  --run runs/federated_scoring_v2 `
  --variant benign `
  --label 0 `
  --profile hybrid
```

The first command defaults to `--label 1`. Labels select a demonstration message; they are not passed to the detector as evidence.

The command selects the first eligible test window for that variant/label, replays earlier observations from its receiver session to build sender and neighbour history, and processes the selected message as the next arrival. This is a replay of an actual dataset message, not a newly generated synthetic message.

The printed output includes receiver, sender, message ID, individual errors, calibrated evidence, final score, and decision. A selected attack message can still be missed; its label does not force a Suspicious result.

## 8. Use hybrid in your own prediction script

The earlier prediction example was pasted directly into PowerShell. There is no separate saved prediction file to locate unless you saved one yourself.

For a Python script that you create, use:

```python
from upgrade_scoring import StreamingDetector

detector = StreamingDetector(
    "runs/federated_scoring_v2",
    profile="hybrid"
)

# messages must be your chronological observable message dictionaries,
# all belonging to ONE receiver/session.
for message in messages:
    result = detector.update(message)
    print(result)
```

This integration example assumes your script supplies `messages`. Place the script in the project root so it can import `upgrade_scoring.py`.

Each message needs these fields:

| Fields | Format |
| --- | --- |
| `rx_time`, `tx_time` | Seconds |
| `pseudonym`, `message_id` | Sender pseudonym and message identifier |
| `x`, `y`, `rx_x`, `rx_y` | Sender/receiver positions in the same local metre coordinate system |
| `speed` | Metres per second |
| `acceleration` | Metres per second squared |
| `heading` | Degrees clockwise from north |

Keep one detector instance per receiver/session and feed messages in reception-time order. Do not create a new detector for every message. With the current history length of 10, a sender needs its previous 10 observations before its next message can be scored. Warm-up returns **Uncertain** with `reason="insufficient_history"`.

The result's `raw_errors` and `calibrated_evidence` align with `evidence_names`. `stream_mse` contains the legacy three errors for inspection; it is not the hybrid fusion rule.

Inference updates message history, not neural weights or cloud weights. Federated model updates happen during a separate training process.

## 9. What hybrid actually changes

The trained movement GRU, timing GRU autoencoder, and context MLP autoencoder are retained.

Movement errors for `dx`, `dy`, speed, acceleration, sine of heading, and cosine of heading are kept separate. Each error is compared with its own benign reference distribution. Timing and context contribute their reconstruction errors.

The additional physical check estimates current speed as:

```text
interval speed = sqrt(dx² + dy²) / elapsed send time
estimated current speed = interval speed + 0.5 × reported acceleration × elapsed send time
endpoint error = |reported current speed − estimated current speed|
```

This assumes approximately constant acceleration over the interval. Position noise, turning, or false acceleration can affect it. It does not require road-map data and does not introduce road data into the current implementation.

Hybrid takes the maximum of the nine calibrated evidence values. Calibration uses an empirical upper-tail transformation with a logarithmic extension beyond the largest reference error, so extreme errors do not all receive an identical capped rank. These scores are **not attack probabilities**.

The low/high thresholds use the saved run's quantiles, normally 0.95/0.99, fitted on separate benign calibration windows:

- Score above the high threshold: **Suspicious**.
- Score above the low threshold, but not above the high threshold: **Uncertain**.
- Otherwise: **Trusted**.

An invalid time interval makes hybrid return Uncertain rather than assigning zero consistency error. Comparison metrics/curves use the same valid windows for every method; all-window decision counts remain available separately.

## 10. Output files

| File inside the evaluation folder | Contents |
| --- | --- |
| `predictions.csv.gz` | All test-window predictions; primary `score` and `decision` columns are hybrid when `--profile hybrid` is used |
| `metrics_comparison.csv` | Overall and per-variant metrics for each method and policy |
| `decision_counts_all_windows.csv` | Counts of Trusted/Uncertain/Suspicious outcomes, including invalid-interval cases |
| `evaluation.json` | Input paths, model/scoring hashes, selected profile, and evaluated-window counts |
| `figures/*.png` | Display-ready ROC and precision–recall figures |
| `figures/*.pdf` | Vector versions of the figures |

The comparison file contains original methods too. Always filter `method == "hybrid"` when reporting the current detector's metrics.

## 11. Common issues

### The PowerShell window was closed

`$hybridResults` is a temporary variable. Find your saved output folder:

```powershell
Get-ChildItem .\runs -Directory -Filter "hybrid_results_*" |
  Sort-Object Name -Descending |
  Select-Object Name
```

Restore the variable using the exact folder you want, for example:

```powershell
$hybridResults = "runs/hybrid_results_REPLACE_WITH_YOUR_TIMESTAMP"
```

Then run the metric or graph commands again. Do not rerun evaluation just to open saved results.

### An output folder already exists

Evaluation refuses to overwrite it. Rerun Step 4 to obtain a fresh timestamped folder. If calibration failed after creating a partial folder, use a new calibration output name and use that name consistently in subsequent commands.

### `scoring_upgrade.json` is missing

Complete Step 3 successfully. Point prediction/evaluation at `runs/federated_scoring_v2`, not the original model folder.

### Results still use the old scoring

Use `upgrade_scoring.py evaluate --profile hybrid` and `from upgrade_scoring import StreamingDetector`. The old `from fedcontext.inference import StreamingDetector` continues to use the original scoring.

## 12. Reporting results in the paper

Report the measured hybrid AUC, recall, and false-alarm rate together, both overall and per attack family. Improvement is not guaranteed for every family. Constant-position offsets remain a known weakness because a consistent shift can preserve displacement and speed.

The endpoint-only diagnostic result is not the full hybrid result. Use this workflow's hybrid rows and curves for claims about the combined detector.

Because the scoring design was revised after examining these test results, treat this dataset as development evidence. Confirm the final design on untouched sessions before making final generalisation claims. Keep the original model/results as the baseline for the comparison.
