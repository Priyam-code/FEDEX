# Verified network demo (synthetic only)

One aggregation server and three RSU processes exchanged model parameters over actual localhost HTTP. Two rounds completed. All clients received exactly the same final model. The model was calibrated and evaluated separately, and loaded for streaming inference.

These files are functional evidence, not VeReMi accuracy or a physical/cloud deployment. No raw data or authentication credentials are included. The model in `deployable_model` was trained only on synthetic demo traffic. Do not use its numbers as real-dataset research results.

See `VERIFIED_NETWORK.json`, the logs and CSV files. Reproduce with `python scripts/network_smoke_test.py --out runs/network_smoke`.
