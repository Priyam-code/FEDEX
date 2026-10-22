# Experiment results



Dataset experiment: review audits, configurations and limits before publication.



## Main strict-policy metrics



          method  f1_mean  fpr_mean  roc_auc_mean  average_precision_mean
     centralized 0.032542  0.026538      0.454260                0.085702
       federated 0.059770  0.024593      0.487922                0.097274
isolation_forest 0.019210  0.013593      0.493458                0.094169
   one_class_svm 0.139171  0.078448      0.543518                0.113509



## How to interpret



Strict: only Suspicious is positive; Uncertain is negative. Alert: Uncertain and Suspicious are positive. Neither policy silently removes uncertain cases.

Results are on the current message label at each complete history window. Warm-up messages are excluded; inspect coverage.json. Scores are ranks, not calibrated attack probabilities.

AP and trapezoidal PR-AUC are separate measures. Undefined metrics are blank/null, never fabricated as zero.

Pool metrics are descriptive: attack archives can share benign traces. Use per_scenario_attack.csv for attack-family claims and do not treat all windows as independent observations.

Seed standard deviations reflect training randomness only, not independent traffic simulations. Missing std for one seed is intentional.

## Included outputs



Each neural run has models, calibration, predictions, seven stream-subset ablations, per-attack tables, confusion matrices, ROC/PR curves, learning curves, episode delays, payload communication cost and measured CPU inference latency.

The implementation is a single-process federated simulator. No secure aggregation, DP, distributed network, production OBU integration, or automatic safety action is implemented.