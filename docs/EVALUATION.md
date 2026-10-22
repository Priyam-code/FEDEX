# Turning exported files into an honest results section

## First distinguish three things

1. Synthetic demo: checks whether software works. Its generated attacks are simple. Do not put those scores in a paper as VeReMi performance.
2. Real-data pilot: runs the actual pipeline on one dataset scenario/attack subset with short training. Report it as preliminary and retain its limitations.
3. Final experiment: a predeclared protocol with adequate scenarios, seeds, baselines, calibration and efficiency measurements. Run this yourself before making general research claims.

## Files to use

| File | What it supplies |
|---|---|
| `summary.csv` | Per-method/subset/policy mean, standard deviation and valid seed count |
| `metrics.csv` | All seven neural stream subsets under both policies |
| `per_scenario_attack.csv` | Scenario/attack-family precision, recall, F1, FPR and ranking metrics |
| `predictions.csv.gz` | Auditable per-window truth, decision, error and calibrated score |
| `coverage.json` | How many raw messages were actually scored, including class/family coverage |
| `confusion_strict.png`, `confusion_alert.png` | False positives/negatives under the two operating policies |
| `roc_pr.png`, `score_distribution.png` | Ranking curves and score overlap |
| `learning_curve.csv` / `.png` | Benign selection loss across epochs or rounds |
| `latency.json` | Measured CPU batch-1 median/p95 inference-plus-fusion time |
| `training.json` | Parameter count, checkpoint size, steps, runtime and total payload bytes |
| `communication.csv` | Upload/download parameter bytes by FL round |
| `episode_delays.csv` | Attack-labelled episodes, detection status and first scored-message delay |
| `calibration.json` | Score-reference distributions and seven pairs of thresholds |
| `ingestion_audit.json` | Input file hashes, baseline label assumptions, excluded files and errors |
| `results_table.tex` | Small LaTeX main-method table; add seed variability and experiment scope yourself |

## Metrics and operating policies

Attack is the positive class. Precision = TP/(TP+FP), recall = TP/(TP+FN), F1 = 2TP/(2TP+FP+FN), FPR = FP/(FP+TN). A zero denominator is reported as undefined/null, not invented.

Strict policy counts only Suspicious as positive. An attack marked Uncertain is a false negative under this policy. Alert policy counts both Uncertain and Suspicious as positive. A benign Uncertain case is a false positive under this policy. We do not discard uncertain cases and then claim high accuracy. The uncertainty rate and decided coverage are reported separately.

ROC-AUC and precision–recall curves operate on continuous scores. Average precision (AP) is exported separately from trapezoidal PR-AUC: they are related but not identical. When only one truth class is present, these ranking metrics are left undefined.

Labels describe the current received message, not an entire sender's identity. Histories can contain earlier attack-labelled messages. This permits measuring the effect of corrupted history; normal messages following an attack are still evaluated against their current normal label.

## Do not overlook coverage

The first ten messages for a pseudonym, or after a gap reset, cannot be scored by the full movement predictor. Streaming inference returns Uncertain with `insufficient_history`; offline model evaluation excludes these cases and exports coverage. Never call scored-window recall “recall on every message” without explaining warm-up exclusions. Frequent pseudonym changes or short attacks can reduce coverage substantially.

Episode delay starts at the first **scored** attack-labelled message in a consecutive positive-label episode. It is not ground-truth attack-start-to-detection latency. Undetected episodes remain in the file with a missing delay. Report episode detection rate together with latency among detected episodes; never substitute zero for misses. Receiver/pseudonym changes and gaps can split episodes.

## Suggested results tables

Table 1: centralized three-stream vs federated three-stream vs Isolation Forest vs one-class SVM, using strict-policy F1, FPR, ROC-AUC and AP, plus uncertainty and coverage.

Table 2: all seven stream subsets, with the same operating policy. Explain whether timing/context genuinely improve the full model on the chosen dataset; negative findings are valid results.

Table 3: per-attack recall and FPR under both policies, broken down by scenario. Include benign-only scenario FPR separately.

Table 4: neural parameters, checkpoint size, batch-1 median/p95 CPU inference time, total training time and FL payload bytes. State hardware, thread count, rounds and exclusions.

## Fair comparisons and pitfalls

- Use the same scored windows and train/calibration split across methods. Baselines use current movement, movement variability, timing mean/std and current context. They are not exact sequence-model replicas.
- Isolation Forest uses at most the configured benign training sample cap; one-class SVM is capped at 5,000 samples for its computational cost. Actual sample counts are saved. Disclose this asymmetry.
- Research centralized/federated epoch budgets are approximately matched per sample (40 versus 20×2), but early stopping and optimizer behaviour differ. Demo budgets are deliberately shorter and not equal.
- Attack datasets may reuse baseline traces. Grand pooled metrics over many attack archives can repeatedly count benign traffic. Prefer scenario/attack tables and clearly defined macro summaries. Do not claim confidence intervals based on millions of independent windows.
- Repeated seeds reuse the same test traffic; they are not independent dataset replications. Use scenario/run-level resampling if adding statistical confidence intervals.
- Fit thresholds on benign validation only. If test results guide architectural changes, that test set has become development data; reserve a new final holdout.
- Congested roads do not necessarily cause each vehicle to transmit faster. Neighbour count/speed and observed receiver traffic supply context; verify rather than assume the relationship.
- Streamlining an OBU decision does not prevent a radio channel from being saturated. Flooding detection is not flooding prevention.
- Federated payload bytes exclude cryptography, transport, connection setup, validation/calibration sharing and OBU downloads. They are not measured network bandwidth.
- No secure aggregation or differential privacy is implemented. Do not call these results proof of privacy protection.

## Results paragraph template

“We evaluated [methods] on [named scenarios and attack families], preserving the official splits. Models were fitted on attack-free training data and calibrated using separate benign validation receiver groups. Across [N] seeds, the [method] achieved [measured mean ± std] F1 and [measured mean ± std] FPR under the [strict/alert] policy. The uncertainty rate was [value], and [value]% of test messages had sufficient history to be scored. The [named ablation] changed [metric] by [measured amount]. Median and p95 CPU inference time were [values] on [hardware]. These results are limited to [scope].”

Fill only from real measured exports. Do not replace placeholders with hoped-for improvements.
