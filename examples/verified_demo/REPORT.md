# Experiment results



SYNTHETIC DEMO ONLY: do not present these numbers as VeReMi research results.



## Main strict-policy metrics



          method  f1_mean  fpr_mean  roc_auc_mean  average_precision_mean
     centralized 0.610234  0.019432      0.925014                0.876792
       federated 0.058318  0.007940      0.726421                0.562199
isolation_forest 0.771216  0.010781      0.924366                0.906157
   one_class_svm 0.902009  0.104262      0.991847                0.987462



## How to interpret



Strict: only Suspicious is positive; Uncertain is negative. Alert: Uncertain and Suspicious are positive. Neither policy silently removes uncertain cases.

Results are on the current message label at each complete history window. Warm-up messages are excluded; inspect coverage.json. Scores are ranks, not calibrated attack probabilities.

AP and trapezoidal PR-AUC are separate measures. Undefined metrics are blank/null, never fabricated as zero.

Pool metrics are descriptive: attack archives can share benign traces. Use per_scenario_attack.csv for attack-family claims and do not treat all windows as independent observations.

Seed standard deviations reflect training randomness only, not independent traffic simulations. Missing std for one seed is intentional.

## Included outputs



Each neural run has models, calibration, predictions, seven stream-subset ablations, per-attack tables, confusion matrices, ROC/PR curves, learning curves, episode delays, payload communication cost and measured CPU inference latency.

The implementation is a single-process federated simulator. No secure aggregation, DP, distributed network, production OBU integration, or automatic safety action is implemented.