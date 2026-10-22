"""Contract checks for the simplified extracted-JSON reader."""
import copy
import json
import pytest
from fedcontext.ingest import ingest, parse_message, raw_files, records, vector
from fedcontext.utils import dump_json


def message():
    return dict(rcvTime=1001, sendTime=1000, sender_alias="a", messageID="1", attacker=0,
                sender=dict(pos=[1, 2, 0], spd=10, acl=0, hed=450), receiver=dict(pos=[0, 0]))


def manifest(tmp_path, **source):
    path = tmp_path/"manifest.json"
    dump_json(path, dict(origin="custom", timestamp_unit="ms", sources=[
        dict(path="raw", role="baseline", **source)]))
    return path


@pytest.mark.parametrize("value", [[[1, 2]], [1], [1, 2, 3, 4], [1, float('inf')]])
def test_invalid_position_dimensions(value):
    with pytest.raises(ValueError):
        vector(value)


@pytest.mark.parametrize("key,value", [("sender_alias", None), ("sender_alias", []),
                                      ("messageID", ""), ("messageID", True)])
def test_invalid_identity(key, value):
    obj = message()
    obj[key] = value
    with pytest.raises(ValueError, match=key):
        parse_message(obj)


@pytest.mark.parametrize("entry", [None, 4, [], "message"])
def test_non_object_entries_are_audited(tmp_path, entry):
    dump_json(tmp_path/"raw"/"Train"/"veh.json", [entry])
    with pytest.raises(ValueError, match="Invalid records"):
        ingest(manifest(tmp_path), tmp_path)
    audit = json.loads((tmp_path/"ingestion_audit.json").read_text())
    assert audit['invalid_records'] == 1
    assert not (tmp_path/"messages.csv.gz").exists()


def test_label_assumption_requires_boolean(tmp_path):
    with pytest.raises(ValueError, match="JSON boolean"):
        ingest(manifest(tmp_path, assume_benign="false"), tmp_path)


def test_jsonl_extension_rejected_even_alongside_json(tmp_path):
    dump_json(tmp_path/"veh.json", [message()])
    (tmp_path/"veh.jsonl").write_text(json.dumps(message()))
    with pytest.raises(ValueError, match="JSON arrays only"):
        raw_files(tmp_path)


def test_record_duplicates_are_preserved(tmp_path):
    dump_json(tmp_path/"raw"/"Train"/"veh.json", [message(), message()])
    frame, _ = ingest(manifest(tmp_path), tmp_path)
    assert len(frame) == 2
    assert frame.message_id.tolist() == ['1', '1']
    assert frame.heading.tolist() == [90, 90]
    assert frame.rx_time.tolist() == pytest.approx([1.001, 1.001])


def test_reformatted_message_cross_split_leakage(tmp_path):
    obj = message()
    dump_json(tmp_path/"raw"/"Train"/"veh.json", [obj])
    obj2 = copy.deepcopy(obj)
    obj2['unrelated_metadata'] = 'different file hash'
    dump_json(tmp_path/"raw"/"Test"/"veh.json", [obj2])
    with pytest.raises(ValueError, match="Message identity overlap"):
        ingest(manifest(tmp_path), tmp_path)


def test_bom_and_trailing_garbage():
    assert list(records(b'\xef\xbb\xbf' + json.dumps([message()]).encode())) == [message()]
    with pytest.raises(ValueError):
        list(records(json.dumps([message()]).encode() + b'garbage'))
