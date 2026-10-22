import json
import pytest
from fedcontext.ingest import ingest
from fedcontext.utils import dump_json


def raw(label_marker=None):
    r = dict(rcvTime="1000001", sendTime="1000000", sender_alias="123", messageID="1",
             sender=dict(pos="1,2,0", spd="10", acl="0", hed="90"), receiver=dict(pos="0,0,0"))
    if label_marker is not None:
        r["attacker"] = label_marker
    return r


def test_explicit_baseline_missing_labels(tmp_path):
    dump_json(tmp_path/"base"/"Train"/"veh.json", [raw()])
    m = tmp_path/"manifest.json"
    dump_json(m, dict(origin="custom", timestamp_unit="ns", sources=[dict(path="base", role="baseline", assume_benign=True)]))
    frame, audit = ingest(m, tmp_path)
    assert frame.label.tolist() == [0]
    assert audit["manifest_assumed_benign_records"] == 1
    assert frame.rx_time.iloc[0] == pytest.approx(.001000001)


def test_missing_attack_labels_rejected(tmp_path):
    dump_json(tmp_path/"attack"/"Test"/"veh.json", [raw()])
    m = tmp_path/"manifest.json"
    dump_json(m, dict(origin="custom", timestamp_unit="ns", sources=[dict(path="attack", role="attack", variant="test_attack")]))
    with pytest.raises(ValueError, match="Invalid records"):
        ingest(m, tmp_path)
    assert (tmp_path/"ingestion_audit.json").exists()
    assert not (tmp_path/"messages.csv.gz").exists()


def test_exact_file_cross_split_leakage_rejected(tmp_path):
    for split in ("Train", "Test"):
        dump_json(tmp_path/"base"/split/"veh.json", [raw(0)])
    m = tmp_path/"manifest.json"
    dump_json(m, dict(origin="custom", timestamp_unit="ns", sources=[dict(path="base", role="baseline")]))
    with pytest.raises(ValueError, match="Identical file content across splits"):
        ingest(m, tmp_path)


def test_metadata_never_becomes_model_input(tmp_path):
    r = raw(0)
    r["sender"]["distance_to_road_edge"] = 999999
    r["sender"]["driversProfile"] = "ATTACK"
    r["sender_id"] = "hidden_ground_truth_identity"
    dump_json(tmp_path/"base"/"Train"/"veh.json", [r])
    m = tmp_path/"manifest.json"
    dump_json(m, dict(origin="custom", timestamp_unit="ns", sources=[dict(path="base", role="baseline")]))
    frame, _ = ingest(m, tmp_path)
    assert not {"sender_id", "driversProfile", "distance_to_road_edge"}.intersection(frame.columns)


def test_moving_dataset_does_not_change_record_ids(tmp_path):
    import shutil
    left = tmp_path/"left"
    right = tmp_path/"different_parent"
    dump_json(left/"base"/"Train"/"veh.json", [raw(0)])
    dump_json(left/"manifest.json", dict(origin="custom", timestamp_unit="ns", sources=[dict(path="base", role="baseline")]))
    shutil.copytree(left, right)
    a, _ = ingest(left/"manifest.json", left)
    b, _ = ingest(right/"manifest.json", right)
    assert a.record_id.tolist() == b.record_id.tolist()
    assert a.session.tolist() == b.session.tolist()


def test_csv_gzip_complete_roundtrip(tmp_path):
    import pandas as pd
    from fedcontext.utils import save_csv
    a = pd.DataFrame({"value":range(10000), "message":["test"]*10000})
    p = tmp_path/"records.csv.gz"
    save_csv(a, p)
    pd.testing.assert_frame_equal(a, pd.read_csv(p))
    assert not (tmp_path/"records.csv.gz.partial").exists()
