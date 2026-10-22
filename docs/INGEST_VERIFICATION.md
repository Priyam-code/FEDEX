# Simplified ingestion verification

Verified on 7 October 2026.

Only `src/fedcontext/ingest.py` changed among application Python modules. This was checked by comparing file bytes against the delivered v1 project. Supporting documentation, the manifest explanatory note, tests and package hashes were updated.

## Automated tests

41 tests passed. The two previous tests that accepted nested ZIPs and flexible JSON containers now verify that unsupported raw formats are rejected. The remaining original tests were retained. Added tests cover strict positions and identifiers, invalid message entries, boolean-only manifest assumptions, JSONL rejection, repeated-message preservation, malformed JSON and cross-split identity leakage despite different file formatting.

Run from the project root after installing the development dependencies:

```powershell
python -m pytest -q
```

## Old/new data comparison

The old and new ingestion functions each processed the same extracted input files through the unchanged `prepare.py`. All four feature arrays were compared for exact equality, not approximate similarity. Message tables, window metadata and ingestion audits were also compared.

| Check | Synthetic demo | Real VeReMi NextGen pilot |
|---|---:|---:|
| Raw messages | 30,660 | 144,196 |
| Prepared windows | 26,460 | 71,148 |
| Movement, target, timing, context arrays | Exactly equal | Exactly equal |
| Window metadata | Exactly equal | Exactly equal |
| Canonical message tables | Exactly equal | Exactly equal |
| Ingestion audits | Equal | Equal |

The real-data comparison used the available highway_2 baseline and constantPositionOffset pilot. This does not prove compatibility with every dataset or attack export. The new raw-format contract applies to every new input.

The real-pilot partition counts were unchanged: 18,401 train; 4,542 selection; 3,016 score reference; 2,525 calibration; 42,664 test.

The unchanged suite also exercises model forward/backward passes, FedAvg, calibration and saved-model streaming/offline parity. Models were not retrained for this ingestion refactor: exact input equality and unchanged model/training modules establish the relevant compatibility check. Existing bundled experiment results retain their original provenance and limitations.

See `INGEST_VERIFICATION.json` for the machine-readable comparison summary. The test command uses the same installed dependencies as the project, including CPU PyTorch.
