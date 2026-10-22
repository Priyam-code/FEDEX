# Start here

This revision uses the simplified extracted-JSON reader. Read `docs/INGEST_FORMAT.md` before preparing real data. If updating an existing installation, only the application file `src/fedcontext/ingest.py` needs replacing.

1. Extract the ZIP and open the `FedContextTrust_V2X` folder in VS Code.
2. Follow **README.md → Windows installation**. Do not replace your old Person 1 folder.
3. Run the synthetic demo first. It exercises the full code without downloading data.
4. Look in `examples/verified_demo/` and `examples/real_pilot/` for already-run outputs.
5. Follow the real-data commands in README.md to run your own experiment.
6. Read `docs/EVALUATION.md` before copying numbers into a research paper.

The pipeline runs, but “runs successfully” does not mean “achieves good detection”. The bundled real-data pilot is weak on constant-position-offset attacks. Use its outputs as preliminary research evidence, not a finished success claim. Secure aggregation and actual OBU deployment are not included.

For actual parameter transfer between independent RSU and server processes, read `docs/NETWORK_FEDERATION.md`. Start with `python scripts/network_smoke_test.py --out runs/network_smoke`. The original commands still use the original simulation.
