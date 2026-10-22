# Network implementation verification

Verified on 9 October 2026. This is a networked research prototype; no paid cloud service or physical RSU was deployed.

## Tests

`python -m pytest -q`: 47 passed in 3.86 seconds in the verification environment.

The original ingestion, features, models, calibration and inference tests remain. New tests verify finite numeric model transport, parameter schema rejection, unequal-sample FedAvg, round/base-model/run-identity checks, idempotent retries, authentication failures, real HTTP upload/download, explicit remote-HTTP opt-in, missing-client timeout, and HTTPS transfers with a locally generated and verified test certificate.

## Separate-process integration test

`python scripts/network_smoke_test.py --out runs/network_smoke` successfully:

1. Generated and prepared 30,660 synthetic messages into 26,460 windows.
2. Distributed 3,300 benign training windows into three local client shards of 1,650, 1,375 and 275 windows.
3. Started an aggregation service and three separate Python RSU processes.
4. Completed two rounds through localhost HTTP, accepting six model uploads.
5. Confirmed the downloaded final model hash and every tensor exactly matched the server at all three RSUs.
6. Calibrated the final model from held-out benign partitions, exported test metrics and loaded the calibrated model for streaming inference.

Observed model upload bodies totalled 321,516 bytes. Model download bodies totalled 482,274 bytes (including final downloads). These are measured application-body sizes; HTTP/TLS/TCP overhead and status polling are excluded. The test used HTTP; HTTPS is covered separately by the transport test.

Evidence is included under `examples/network_demo/`. No authentication tokens, private keys or raw datasets are included. The synthetic trained checkpoint is clearly labelled and must not be reported as real VeReMi performance.

## Compatibility and limits

The existing application Python files were preserved unchanged; the network package adds a new command module. The network mode reuses `ThreeStream`, `local_train`, `fedavg`, calibration and evaluation functions. It uses fixed rounds and a final-round checkpoint, unlike the original simulation's selection/early stopping, and it is not claimed to reproduce the old training weights bit for bit.

The code supports remote addresses and TLS, but different physical machines and a public cloud were not tested in this environment. Remote deployment instructions are included. Server-crash recovery, secure aggregation, differential privacy, poisoning resistance and automatic OBU provisioning remain outside this implementation.
