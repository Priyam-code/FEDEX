# Security boundary and remaining work

Implemented: benign-only neural training, virtual-client partitions, parameter averaging, causal scoring, static input validation and held-out evaluation. The coordinator is simulated within the same process and can access experiment data. This is an algorithm experiment, not an isolation/security proof.

The added network mode runs independent RSU/client processes and an HTTP(S) aggregation server. It implements per-client bearer authentication, optional certificate-verified TLS, strict numeric update validation and round/retry checks. The server receives model weights; its code does not load client feature files. This has been tested on localhost, not on physical RSUs or a public cloud. Offline benchmark partitioning and post-training calibration still use a trusted evaluation machine.

Not implemented: secure aggregation, differential privacy, Byzantine/poisoning-resistant aggregation, physical/public-cloud deployment, V2X certificate validation or access to real OEM traces. Authentication and optional encrypted transport are available only in the new network mode.

Model parameters can reveal information about training data. Hiding raw uploads is helpful in an architecture, but does not automatically make updates private. Calibration score sharing and preprocessing also need a privacy design. Do not describe plain FedAvg as “securely aggregating” parameters in a paper about this release.

Trust assumptions: baseline data are actually clean; the manifest is correct; receiver clock/position are usable; a supplied map is aligned; neighbour observations are not universally colluding; and the model is distributed from a trusted source. Sender content itself can be malicious, so neighbours and claimed position are consistency evidence rather than independent ground truth.

The default has no road map. The map adapter is a nearest-segment baseline, not a full lane matcher. Missing timing information is not fabricated. Hardware inference time is not end-to-end safety latency. Anomaly detection is not intrusion prevention or safe vehicle control.

Warm-up excludes many short tracks; this is prominent in the real pilot. Future work should evaluate cold-start policies, shorter histories and coverage/accuracy trade-offs using development data before a new final test. Pseudonym changes can also weaken longitudinal trust; persistent identity linking is neither assumed nor implemented.

Benign-only training still needs trustworthy normal data. This package does not solve contamination, concept drift, poisoned RSUs, colluding neighbours or arbitrary unknown attacks. Operational deployments must account for privacy, security and safety engineering beyond this prototype.
