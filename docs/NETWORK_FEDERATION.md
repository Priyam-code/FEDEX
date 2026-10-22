# RSU-to-server communication: working research implementation

The project now has two training modes:

- `python -m fedcontext run ...` retains the original single-process simulation and centralized baseline.
- `python -m fedcontext.network ...` runs a server and independent RSU processes that exchange parameters through real HTTP(S) requests.

The second mode can run in separate terminals on one computer or on different machines. A cloud VM can host the server, but no cloud account or deployment is created by this package. The verified run used localhost processes, not physical RSUs or a public cloud.

## What each machine does

| Machine/process | Receives | Does locally | Sends |
|---|---|---|---|
| RSU client | Current global weights and round metadata | Trains all three models on its own benign feature windows with Adam | Model parameters in numeric NPZ format |
| Aggregation server | Client weights; sample counts are provisioned in its setup manifest | Validates each update, runs sample-weighted FedAvg, saves the global model | New global weights |
| Trusted evaluation machine | Completed global checkpoint | Fits score calibration on held-out benign data and evaluates test data | A deployable model directory, distributed separately |

The server code does not open a training-data file. Its setup folder contains only configuration, client sample counts and authentication tokens. The initial `partition` command reads a prepared benchmark centrally to simulate distributing regional data; that is offline experiment setup, not a claim that the original dataset has never been centrally stored.

## Round flow

1. Every RSU authenticates with its own bearer token and checks server configuration and experiment identity.
2. Each RSU downloads exactly the same round's model through `GET /model`.
3. It trains with its local `train.npz` and the existing `local_train()` function.
4. It uploads parameters through `POST /update`, including the round, server-run ID and base-model hash.
5. The server waits for all configured clients, then calls the existing `fedavg()` function with their fixed sample counts.
6. The next status response advertises the new model. Clients download it for the next round.
7. After the last round, every client downloads the final global weights and exits. Stop the server with Ctrl+C after all clients finish.

Clients do not send messages, feature windows, labels or local training losses over the network. Local losses stay in each client's output folder. Model weights may still reveal information; neither FedAvg nor TLS provides differential privacy.

## Quick automatic demonstration

From the extracted project root, with the project installed in your activated virtual environment:

```powershell
python -m pip install -e ".[dev]"
python scripts/network_smoke_test.py --out runs/network_smoke
```

Use a new output directory each time. This script generates the existing synthetic demo, prepares its data, starts a server and three separate RSU processes, runs two rounds, compares the final model at all clients, calibrates the model, evaluates it and checks streaming inference. It shuts down its server automatically.

Open:

- `runs/network_smoke/server.log`
- `runs/network_smoke/client_0.log` (and the other client logs)
- `runs/network_smoke/VERIFIED_NETWORK.json`
- `runs/network_smoke/deployable_model/evaluation/metrics.csv`

These are synthetic functionality results, not real-dataset research results.

## Manual demonstration in separate terminals

### 1. Generate and prepare demo data

```powershell
python -m fedcontext generate-demo --out data/network_demo_raw
python -m fedcontext prepare --manifest data/network_demo_raw/manifest.json --config configs/demo.yaml --out data/network_demo_prepared
python -m fedcontext.network partition --prepared data/network_demo_prepared --config configs/demo.yaml --out data/network_shards --seed 7
```

The supplied demo configuration uses three client slots and three rounds. It creates `server/federation.json` and a `client_N` directory for each nonempty client. Check `data/network_shards/distribution.json` for the actual client IDs. The automatic smoke test deliberately uses two rounds to finish quickly.

### 2. Terminal 1: aggregation server

```powershell
python -m fedcontext.network server --manifest data/network_shards/server/federation.json --out runs/network_server --port 8080
```

The terminal remains running. It prints a message after each successful aggregation.

### 3. Terminals 2, 3 and 4: RSU clients

Run one command per terminal, activating the same virtual environment in each:

```powershell
python -m fedcontext.network client --shard data/network_shards/client_0 --server http://127.0.0.1:8080 --out runs/network_rsu0
```

```powershell
python -m fedcontext.network client --shard data/network_shards/client_1 --server http://127.0.0.1:8080 --out runs/network_rsu1
```

```powershell
python -m fedcontext.network client --shard data/network_shards/client_2 --server http://127.0.0.1:8080 --out runs/network_rsu2
```

All configured clients must participate in each round. A missing client causes a timeout, not a silent partial aggregation. Increase server `--round-timeout` from its 600-second default when clients train slowly.

### 4. Calibrate and evaluate

After every client reports its final model download, stop the server with Ctrl+C. On the trusted evaluation machine:

```powershell
python -m fedcontext.network finalize --prepared data/network_demo_prepared --server-run runs/network_server --out runs/network_deployable
```

This creates `model.pt`, `run.json` and `calibration.json` together. Evaluation outputs are under `runs/network_deployable/evaluation/`.

`finalize` does not retrain. It fits calibration from only `score_ref` and `calibration` benign partitions, then evaluates the held-out test partition. Calibration is currently an offline centralized step; it is not performed by the network aggregation service.

### 5. Test messages using the resulting model

Use the existing canonical one-receiver JSONL format documented in `INFERENCE.md`:

```powershell
python -m fedcontext predict-stream --run runs/network_deployable --input data/one_receiver.jsonl --out runs/network_predictions.jsonl
```

Raw `network_server` and `network_rsuN` checkpoints contain weights but do not yet contain calibrated decision thresholds. Use the `network_deployable` folder for Trusted/Uncertain/Suspicious inference. The first messages build the sender history and return Uncertain until enough history exists.

## Use real VeReMi NextGen data

First prepare extracted raw JSON using `docs/INGEST_FORMAT.md`:

```powershell
python -m fedcontext prepare --manifest configs/veremi_manifest.example.json --config configs/research.yaml --out data/network_real_prepared
python -m fedcontext.network partition --prepared data/network_real_prepared --config configs/research.yaml --out data/network_real_shards --seed 7
```

Start the server with `data/network_real_shards/server/federation.json`. Start one client for every `client_N` directory listed in `distribution.json`. Use fresh run folders and finalize against `data/network_real_prepared`. The research configuration allows five slots; some may be empty. At least two nonempty clients are required.

To repeat for seeds 19 and 43, partition again with those seeds into separate folders and run separate experiments. Do not claim three independent traffic datasets merely because training seeds differ.

## Run on different computers

1. Install this same project and dependencies on the server and RSU machines.
2. Transfer only the `server` setup folder to the aggregation machine.
3. Transfer only `client_0` to RSU 0, `client_1` to RSU 1, and so on. Each contains its own token and local data. Do not send the entire shards directory to every machine.
4. Use a reachable server IP/DNS address. `127.0.0.1` on a client always means that client's own computer.
5. Allow the chosen server port through the host firewall only for the intended clients.

For a trusted local test network, explicitly allow unencrypted HTTP on both sides:

```powershell
python -m fedcontext.network server --manifest server/federation.json --out runs/server --host 0.0.0.0 --port 8080 --allow-insecure-http
```

Example client, replacing `192.168.1.10` with the actual server address:

```powershell
python -m fedcontext.network client --shard client_0 --server http://192.168.1.10:8080 --out runs/rsu0 --allow-insecure-http
```

The explicit flag is for a trusted test LAN. HTTP exposes bearer tokens and weights to network observers; use HTTPS on untrusted networks.

## HTTPS / a cloud-hosted server

The same server supports a TLS certificate and private key:

```powershell
python -m fedcontext.network server --manifest server/federation.json --out runs/cloud_run --host 0.0.0.0 --port 8443 --cert server-cert.pem --key server-key.pem
```

The certificate must be valid for the hostname or IP used by clients. Obtain and install it for your actual server. A certificate trusted by the operating system needs no extra client flag. For a private test CA, provide that CA certificate:

```powershell
python -m fedcontext.network client --shard client_0 --server https://YOUR-SERVER-NAME:8443 --ca-file test-ca.pem --out runs/rsu0
```

There is no option that disables certificate verification. Tokens are loaded from files, not pasted into command lines. Keep `federation.json` and `client.json` private and exclude them from public repositories. Distribute credentials through a trusted channel.

Cloud provisioning, DNS, certificate issuance, managed secrets and production service hardening remain deployment work. The included standard-library HTTP service is an experimental service, not a hardened public API. No paid cloud deployment has been performed.

## Files and evidence to show in a viva

| File | Evidence |
|---|---|
| Server `communication.csv` | Completed rounds, participants, training counts, model-body byte counts, wall-clock round duration and global hash |
| Server `network_events.jsonl` | Actual model downloads, accepted uploads, duplicate retries and aggregations |
| Server `training.json` | Completed rounds and final model hash |
| RSU `client_rounds.csv` | Local losses and actual encoded upload/download sizes per round |
| RSU `download.json` | Final global model hash received by that client |
| Finalized `evaluation/metrics.csv` | Detection metrics for all stream subsets and both uncertainty policies |
| Finalized `evaluation/per_scenario_attack.csv` | Metrics grouped by scenario/attack |
| Finalized `evaluation/latency.json` | Local model inference benchmark, not end-to-end network latency |

The model-body counts come from bytes actually handled by HTTP requests/responses. They exclude HTTP, TLS, TCP overhead and status polling. Duplicate-upload bodies and final-model downloads appear separately in the event log; `communication.csv` counts unique accepted updates for each training round. No network-link bandwidth or radio latency is inferred from these values.

## Code map

| File | Main responsibility |
|---|---|
| `network/setup.py` | Split prepared training data into private client shards; provision tokens and fixed sample counts |
| `network/protocol.py` | Encode/decode numeric parameter arrays with key, shape, dtype, size and finiteness checks |
| `network/server.py` | Authenticate clients, enforce rounds, aggregate updates, serve weights and record transfer evidence |
| `network/client.py` | Download weights, call existing local training, upload updates and save final model |
| `network/finalize.py` | Calibrate and evaluate the final global checkpoint on the evaluation machine |
| `network/__main__.py` | Separate CLI; existing `fedcontext` commands still work |
| `scripts/network_smoke_test.py` | Reproducible multi-process localhost demonstration |

## Experimental differences and remaining limits

- Network mode uses a predeclared fixed round count and returns the last round. It does not pool a selection set at the server or use the simulation's early stopping. Keep this distinction explicit in comparisons.
- All configured clients participate synchronously. There is no dynamic selection or dropout-tolerant partial aggregation.
- Local Adam restarts each round, as in the existing simulation. Network-mode client/round seeds are explicit, so do not expect bitwise equality with the old single-process training run.
- Authentication and TLS are implemented. Cryptographic secure aggregation, differential privacy and malicious-update/poisoning resistance are not. Finite, correctly shaped malicious weights can still harm FedAvg.
- Upload retries are idempotent within a live server run. Server-crash recovery is not implemented; start a coordinated fresh run after a server restart.
- The context stream uses neighbours; network mode does not accept road-map-enabled prepared data.
- The final calibrated model can be used by existing local inference. Automatic deployment to physical OBUs is not implemented.
