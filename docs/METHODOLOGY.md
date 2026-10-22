# Methodology: what the code actually does

## Research question

Does combining movement, communication timing and receiver-local context improve anomaly detection over single streams, and how does simulated federated training compare with centralized training under the same data protocol?

This is a hypothesis to test, not an improvement already proven. The system is not a reproduction of any single referenced paper. In particular, its context autoencoder is our implementation choice inspired by context-based reasoning; it is not claimed to be the congestion paper's exact model.

## Data protocol

Use official VeReMi NextGen baseline and attack archives. Keep the provided Train/Validation/Test splits. Baseline archives are the only fitting sources. Their missing labels may be assigned 0 only through the explicit `assume_benign` manifest setting. Attack records require their own labels. No labels, attack names, simulator vehicle IDs, driver profiles or ground-truth road-edge distances become input features.

NextGen has receiver-specific records. Process each file as its own receiver session; sort by receive time. Sender histories use the transmitted pseudonym (`sender_alias`) and reset after a receive gap exceeding five seconds. Preserve repeated messages: deleting them would remove evidence useful for replay/flooding experiments.

Whole baseline Validation receiver files are deterministically assigned to selection, score-reference or threshold-calibration roles using a stable hash of scenario and receiver. This avoids overlapping windows crossing those roles. It does not make correlated nearby traffic statistically independent. Exact file duplication and matching message identities across official splits are checked. Provenance auditing is still necessary when combining separately generated simulation runs.

## Causal features and windows

The movement model observes ten previous movement feature vectors and predicts the current one. Timing uses a ten-message sequence ending at the current received message. Context uses only the current received message, receiver position, a cache of already-received neighbours and an optional external road map. Future messages never supply context.

Neighbour entries expire after one second and are compared within 300 metres of the sender's claimed location. Median neighbour speed and angular disagreement reduce sensitivity to individual neighbours, but they do not make context immune to collusion or Sybil attacks. Geometric proximity is not the same as same-lane membership.

Fixed physical scaling constants are shared by all clients. They are not fitted to pooled raw data or test data. No statistical standardization is silently performed across private RSU records. Constants are starting choices and should be frozen before final evaluation.

## Exact models

Movement: one GRU layer, input dimension 6, hidden dimension 32. Its last hidden state passes through Linear(32,16), ReLU and Linear(16,6). The anomaly error is mean squared error between the predicted and observed current six-dimensional movement state.

Timing: GRU encoder with input dimension 7 and hidden dimension 24. Repeat the encoded state across the sequence; a GRU decoder with input/hidden dimension 24 reconstructs it, followed by Linear(24,7). Average squared reconstruction error over time and features.

Context: fully connected autoencoder 10→16→8→4→8→16→10, ReLU after every hidden layer, linear output. Two of the ten inputs indicate neighbour/map availability. This is why the implemented input size is 10 rather than an earlier proposed 8. The anomaly error is mean squared reconstruction error.

The streams have separate parameters. Their three mean losses are summed for optimizer updates. Adam learning rate is 0.001; minibatch size 128; gradient norm is clipped to 5. Research configuration permits 40 centralized epochs or 20 FL rounds with two local epochs per round. A five-step patience rule on the sum of benign selection losses restores the best checkpoint. These are initial configurations, not tuned optimal settings.

## Federated experiment

The research configuration assigns endpoint windows to five virtual RSU buckets using a hash of the receiver's 1 km spatial grid cell and scenario. Using receiver position prevents attacker-claimed location from choosing the client. The demo uses receiver-based hashing. Explicit manifest `rsu` values can also define assignment keys. Only nonempty clients participate; actual counts are exported.

This is a proxy for regional ownership, not a simulation of radio coverage or physical RSU handovers. All clients are simulated on one machine. In each round, each active client gets the same current model, performs local Adam training on its benign windows and returns parameters. The server averages parameter tensors with weights proportional to the number of training windows at each client. Client Adam state resets each round. The centralized optimizer persists across epochs.

Validation and calibration are coordinated offline in the simulator. Their score/statistic sharing has not been protected with secure aggregation. Plain FedAvg leaves model-update leakage and poisoning risks unresolved. Raw-data locality is an architectural intent; this single-process experiment is not evidence of a deployed privacy guarantee.

## Score calibration and decision

Fit an empirical distribution separately for each stream's errors on the benign score-reference partition. Convert each new error to its rank in that distribution. These numbers are not probabilities that a message is malicious.

Take the mean of the available experiment's selected stream scores. For the full method, all three have equal weight. On the separate benign threshold-calibration partition, set low/high thresholds to the 95th/99th percentiles. Above the high threshold is Suspicious; above only the low threshold is Uncertain; otherwise Trusted. Comparisons are strictly greater than thresholds. Ties, finite samples and distribution shifts mean the requested percentiles do not guarantee a particular test false-alarm rate.

The seven stream-subset ablations receive their own calibrated thresholds. These are fusion ablations with independently trained streams, not seven newly optimized architectures. Equal averaging can dilute a single strong stream; ablations help expose this rather than assuming the combination must win.

“Uncertain” is a reject/grey-zone decision, not a third ground-truth attack class, Bayesian uncertainty estimate or proof of benign behaviour. No automated punishment, braking, blocking or persistent reputation update is implemented.

## Reproducibility

Record dataset hashes, manifest, feature settings, package versions, random seeds, client counts, selected epochs/rounds and calibration thresholds. Compare methods on identical held-out windows. Run at least three training seeds; use independent traffic runs/scenarios when available. Seed standard deviations quantify optimizer randomness, not all sources of research uncertainty.

Full generalization claims need experiments beyond this delivered pilot, including unseen scenarios and realistic benign congestion. Attack labels are not used in training or threshold calibration; all tested attack families are unseen by the anomaly-fitting procedure, but new attacks that resemble normal data may still go undetected.
