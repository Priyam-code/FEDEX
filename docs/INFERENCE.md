# Using the local detector

The class below runs inference on CPU and preserves causal state for one receiver session. It is an interface for later integration with a message receiver, not an actual OBU deployment.

```python
from fedcontext.inference import StreamingDetector

detector = StreamingDetector("runs/real_experiment/federated_seed7")

observation = {
    "rx_time": 12.001,
    "tx_time": 12.0,
    "pseudonym": "12345",
    "message_id": "800",
    "x": 120.0, "y": 20.0,
    "rx_x": 110.0, "rx_y": 18.0,
    "speed": 15.0,
    "acceleration": 0.1,
    "heading": 90.0
}
result = detector.update(observation)
print(result)
```

The first ten messages for a pseudonym return `decision="Uncertain"`, `score=None`, `reason="insufficient_history"`. Later results include three stream errors, three rank scores and the final decision. Never mix receiver sessions in one instance. Keep messages in reception order. Timestamp units are seconds; positions are local metres; headings are degrees clockwise from north. No label is required or used for inference.

For a JSONL file containing these records:

```powershell
.\.venv\Scripts\python.exe -m fedcontext predict-stream --run runs/real_experiment/federated_seed7 --input data/one_receiver.jsonl --out runs/stream_predictions.jsonl
```

If trained with a road map, supply a correctly aligned map with `--road-map`. Receiver and pseudonym identity resolution, signed-message verification, sender authenticity, clock correction and communication ingestion remain the integrating application's responsibility.

Long gaps reset histories. Anomaly decisions do not currently update a persistent sender reputation; “trust” is a per-window decision label. A changed pseudonym starts a new history. The software intentionally performs no blocking, braking or other safety-critical action.

Only load model/calibration files you trust. PyTorch state dictionaries are loaded with `weights_only=True`. Saved classical baseline `.joblib` files use pickle-based serialization and must never be loaded from an untrusted sender.
