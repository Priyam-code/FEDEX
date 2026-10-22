# Simplified ingestion: required raw-data format

The only application module changed is `src/fedcontext/ingest.py`. The command-line interface, feature extraction, window preparation, models, training, calibration, inference and evaluation modules are unchanged.

This reader deliberately supports one file layout. Extract the dataset before running `prepare`. It still accepts the comma-separated positions and numeric strings found in the VeReMi NextGen files tested with this project. You do not need to rename their message fields or rewrite their JSON.

## What became simpler?

Archive recursion and automatic JSON-format detection have been removed. The reader now follows one path: find extracted JSON files, read their message arrays, validate each message, and save the canonical table. `parse_message()` isolates the field conversion from file discovery and provenance checks.

The label checks and checks against train/test leakage remain. Removing those would make the code shorter but could invalidate an experiment.

## Enforced constraints

| Item | Required format or behaviour |
|---|---|
| Raw files | Extracted files with a `.json` extension. ZIP, JSONL and common compressed/archive extensions inside a source are rejected. Other extensions are not treated as message data. |
| JSON layout | One UTF-8 JSON array per file: `[{...}, {...}]`. A UTF-8 BOM is accepted. A lone object, JSONL, or an object wrapping a `messages`/`records`/`data` list is rejected. |
| File size | Each processed JSON file must be at most 256,000,000 bytes. |
| Entries | Every array entry must be an object with `sender` and `receiver` objects. |
| Required fields | `rcvTime`, `sendTime`, `sender_alias`, `messageID`, `sender.pos`, `sender.spd`, `sender.acl`, `sender.hed`, `receiver.pos`. |
| Sender/message IDs | Non-empty strings or integers. No lists, objects, booleans or missing IDs. The hidden simulator `sender_id` is never a fallback. |
| Positions | A flat list of exactly two or three finite numeric coordinates, or a comma-separated numeric string. Optional square/round brackets around the string are accepted. Only x and y enter the canonical table. Nested vectors and longer vectors are rejected. |
| Numeric fields | Values must be convertible to finite numbers. Numeric strings are accepted. Receive/send timestamps must be non-negative. Heading is reduced modulo 360. |
| Timestamp unit | The manifest must explicitly specify `s`, `ms`, `us` or `ns`. Output timestamps are converted to seconds. |
| Origin | The manifest must declare `synthetic_demo`, `veremi_nextgen` or `custom`. |
| Source role | Every source must declare `baseline` or `attack`. |
| Label assumption | `assume_benign`, if present, must be a JSON boolean. Only a baseline source may set it to `true`. |
| Labels | Labels must represent 0 or 1; numbers, numeric strings and booleans are accepted. Missing/null labels become 0 only under an explicit baseline assumption. An attack-labelled record in a baseline source is rejected. |
| Splits | An explicit source `split` must be `train`, `validation` or `test`. Otherwise the source-relative path must identify exactly one Train/Validation/Test folder (case-insensitive; `Val` is accepted). Attack train/validation files are excluded before reading messages. |
| Attack family | Recognised from the path for the existing attack names, or supplied as `source.variant`. Unknown attack-family paths require this setting. |
| Leakage | Identical file content across splits and overlapping message identities across splits are rejected. Exact same-split file copies are skipped; repeated messages within a file are retained for replay/timing features. |
| Invalid messages | Row-level errors are collected in `ingestion_audit.json`, then ingestion fails. The reader does not silently repair labels or drop bad rows to produce a dataset. File-format/path/JSON-syntax errors can stop immediately before an audit is written. |

## Dataset organisation you must preserve

Each file must describe one receiver in one simulation session, with a stable receiver filename. Keep the official split folders, scenario folders, receiver filenames and message order. The loader uses the filename as the receiver identifier; it cannot establish physical receiver identity from an arbitrary renamed file. `prepare.py` sorts receive times within a session, so you do not need to sort the raw array yourself.

Coordinates must be in a shared local metre coordinate system, speed in m/s, acceleration in m/s² and heading in degrees (north = 0, clockwise). These are required semantic assumptions; the reader can validate numeric values but cannot prove their physical units or that a declared baseline is genuinely benign.

Scenario metadata is inferred from known highway/urban folder names; otherwise it defaults to `custom`. For a custom dataset, set `source.scenario` explicitly to avoid mixing unrelated experiments. Keep optional `rsu` metadata if your previous manifest used it.

## Example message file

Save a JSON array such as this in a receiver file. Real files need enough messages per sender for the configured history length.

```json
[
  {
    "rcvTime": "1002000000",
    "sendTime": "1000000000",
    "sender_alias": "vehicle_A",
    "messageID": "100",
    "attacker": 0,
    "sender": {
      "pos": "120.0,50.0,0.0",
      "spd": "15.0",
      "acl": "0.2",
      "hed": "90.0"
    },
    "receiver": {"pos": "110.0,50.0,0.0"}
  }
]
```

Extra fields may remain in the raw file. Only the listed observable fields and the evaluation label are selected; simulator metadata is not made into model features.

## How to replace your existing file

1. Back up your existing `src/fedcontext/ingest.py` outside `src/fedcontext`.
2. Copy `src/fedcontext/ingest.py` from this updated project into the same location in your project. No other application `.py` file needs replacing.
3. Fully extract outer and inner dataset ZIPs into a separate staging directory. Copy only the resulting extracted data tree into `data/raw/benign` or `data/raw/attacks`. Keep original ZIPs outside those source directories.
4. Preserve the official Train/Validation/Test directories and the scenario/attack directory names. Do not combine different receiver files into one large JSON file.
5. Check the manifest: paths must now point to extracted directories or individual JSON files, not ZIPs. The existing example manifest already points to the two raw directories.
6. Run preparation into a fresh output directory. Do not overwrite an existing experimental result.

From the project root in PowerShell:

```powershell
python -m fedcontext prepare --manifest configs/veremi_manifest.example.json --config configs/research.yaml --out data/processed_strict
```

Then use the normal training command:

```powershell
python -m fedcontext run --prepared data/processed_strict --config configs/research.yaml --out runs/strict_experiment --seeds 7 19 43
```

An explicit single-source manifest entry can look like this:

```json
{
  "path": "../data/raw/attacks/InTAS_highway_2_constantPositionOffset/Test",
  "role": "attack",
  "split": "test",
  "scenario": "highway_2",
  "variant": "constantPositionOffset",
  "assume_benign": false
}
```

Do not change the timestamp unit just to make the code run. Match the actual dataset unit.

## Functions in the simplified file

| Function | Input | Output |
|---|---|---|
| `vector(value)` | Position list or comma-separated position string | NumPy array containing x and y |
| `label(value, assume_benign=False)` | Raw label and baseline-assumption flag | Integer 0 or 1 |
| `records(body)` | Bytes of a JSON-array file | Iterator over array entries |
| `raw_files(base)` | `Path` pointing to an extracted directory or JSON file | Sorted list of JSON paths (single file remains a one-item list) |
| `parse_message(obj, assume_benign=False)` | One raw message object | Dictionary of canonical observable fields and label; timestamps still in source units |
| `infer_split(name)` | Source-relative file path | `train`, `validation` or `test` |
| `ingest(manifest_path, output)` | Manifest path and output directory | `(frame, audit)`; timestamps in seconds |

`json_members()` has been removed because archives are no longer supported here. No other application module imported it. The two old tests for archive/container support were updated to test the new rejection behaviour.

## Downstream compatibility

`ingest()` still returns the same 21 columns, in the same order:

`record_id, session, receiver, split, scenario, variant, role, pseudonym, message_id, rx_time, tx_time, x, y, rx_x, rx_y, speed, acceleration, heading, label, rsu, source_file`.

It still writes `messages.csv.gz`, `ingestion_audit.json` and `source_manifest.json`. `prepare.py` continues to generate the same movement, target, timing and context arrays.

Saved models and existing prepared datasets do not require conversion. The raw-training format constraint does not change `predict-stream`, which still accepts its documented canonical JSONL input.

Session IDs include the raw file content hash. Extracting bytes unchanged preserves that hash, while rewriting/reformatting JSON changes it. Keep scenario names and receiver filenames consistent. `source_file` naturally changes when moving from an archive-member path to an extracted path.

See `INGEST_VERIFICATION.md` for the checks performed for this update. Existing example model results are historical demonstrations, not new accuracy claims.
