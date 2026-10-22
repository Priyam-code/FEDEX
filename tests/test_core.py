import copy
import io
import json
import zipfile
import numpy as np
import pytest
import torch
from fedcontext.config import DEFAULTS
from fedcontext.ingest import vector, label, infer_split, records, raw_files, ingest
from fedcontext.features import ReceiverFeatures, RoadMap
from fedcontext.models import ThreeStream
from fedcontext.calibration import Calibrator
from fedcontext.training import fedavg
from fedcontext.metrics import evaluate
from fedcontext.utils import dump_json


def observation(i, alias="a", label_value=0):
    return dict(rx_time=i*.1+.001, tx_time=i*.1, pseudonym=alias, message_id=str(i),
                x=i*2., y=0., rx_x=0., rx_y=0., speed=20., acceleration=0., heading=90., label=label_value)


def test_position_formats():
    np.testing.assert_array_equal(vector("1.2,3.4,0"), [1.2, 3.4])
    np.testing.assert_array_equal(vector([1.2, 3.4, 0]), [1.2, 3.4])


def test_labels_fail_closed():
    assert label("1") == 1
    assert label(False) == 0
    assert label(None, True) == 0
    for value in (None, "benign", 2, -1, float("nan")):
        with pytest.raises(ValueError):
            label(value)


def test_split_windows_paths():
    assert infer_split("a.zip!/folder\\Validation\\inner.zip!/veh.json") == "validation"
    with pytest.raises(ValueError):
        infer_split("data/unknown/a.json")


def test_json_array_only():
    obj = {"sender": {}, "attacker": 0}
    assert list(records(json.dumps([obj]).encode())) == [obj]
    for body in (json.dumps({"messages": [obj]}), json.dumps(obj),
                 json.dumps(obj)+"\n"+json.dumps(obj)):
        with pytest.raises(ValueError, match="Expected a JSON array"):
            list(records(body.encode()))


def test_archives_require_extraction(tmp_path):
    p = tmp_path/"data.zip"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("Train/veh.json", '[{"sender": {}}]')
    with pytest.raises(ValueError, match="Extract archives"):
        raw_files(tmp_path)


def test_no_label_leakage():
    a, b = ReceiverFeatures(DEFAULTS), ReceiverFeatures(DEFAULTS)
    for i in range(12):
        fa = a.update(observation(i, label_value=0))
        fb = b.update(observation(i, label_value=1))
    for k in ("movement", "target", "timing", "context"):
        np.testing.assert_array_equal(fa[k], fb[k])
    assert fa["clean_window"] and not fb["clean_window"]


def test_no_future_and_history_target_separation():
    engine = ReceiverFeatures(DEFAULTS)
    for i in range(10):
        assert engine.update(observation(i)) is None
    f = engine.update(observation(10))
    assert f["movement"].shape == (10, 6)
    assert f["timing"].shape == (10, 7)
    before = copy.deepcopy(f)
    engine.update(observation(11, alias="new"))
    np.testing.assert_array_equal(f["context"], before["context"])
    assert f["context"][8] == 0  # no other sender was previously observed


def test_duplicate_is_preserved_and_visible():
    engine = ReceiverFeatures(DEFAULTS)
    for i in range(11):
        f = engine.update(observation(i))
    row = observation(10)
    row["rx_time"] += .0001
    duplicate = engine.update(row)
    assert duplicate["timing"][-1, 5] > f["timing"][-1, 5]


def test_receiver_order_and_reset():
    engine = ReceiverFeatures(DEFAULTS)
    for i in range(11):
        engine.update(observation(i))
    assert engine.update(observation(100)) is None
    with pytest.raises(ValueError):
        engine.update(observation(2))


def test_map_adapter(tmp_path):
    p = tmp_path/"roads.json"
    dump_json(p, {"coordinate_system": "local_meters", "roads": [{"points": [[0, 0], [100, 0]], "bidirectional": False}]})
    distance, heading, valid = RoadMap(p).match(50, 3, 90)
    assert distance == 3 and heading == 0 and valid == 1
    assert RoadMap().match(1, 2, 90) == (0., 0., 0.)


def test_model_shapes_backward():
    model = ThreeStream()
    errors = model.errors(torch.randn(4, 10, 6), torch.randn(4, 6), torch.randn(4, 10, 7), torch.randn(4, 10))
    assert errors.shape == (4, 3)
    errors.mean().backward()
    assert all(p.grad is not None for p in model.parameters())


def test_fedavg_weighted():
    result = fedavg([{"a":torch.tensor([1.])}, {"a":torch.tensor([3.])}], [1, 3])
    assert result["a"].item() == 2.5


def test_calibration_and_roundtrip():
    ref = np.arange(300).reshape(100, 3)/300
    cal = Calibrator().fit(ref, ref+.002, .8, .95)
    assert len(cal.thresholds) == 7
    scores, decisions, _ = cal.predict(np.array([[0., 0., 0.], [5., 5., 5.]]))
    assert list(decisions) == ["Trusted", "Suspicious"]
    restored = Calibrator.from_dict(cal.to_dict())
    np.testing.assert_array_equal(scores, restored.predict(np.array([[0., 0., 0.], [5., 5., 5.]]))[0])


def test_uncertain_policies():
    y, scores, decisions = [0, 0, 1, 1], [.1, .5, .6, .9], ["Trusted", "Uncertain", "Uncertain", "Suspicious"]
    strict = evaluate(y, scores, decisions, "strict")
    alert = evaluate(y, scores, decisions, "alert")
    assert strict["tp"] == 1 and strict["fn"] == 1 and strict["fp"] == 0
    assert alert["tp"] == 2 and alert["fp"] == 1
    assert strict["uncertainty_rate"] == .5


def test_undefined_metrics_are_not_invented():
    result = evaluate([0, 0], [.1, .2], ["Trusted", "Trusted"])
    assert result["roc_auc"] is None and result["recall"] is None and result["precision"] is None


def test_assume_benign_forbidden_for_attack_manifest(tmp_path):
    manifest = tmp_path/"manifest.json"
    dump_json(manifest, dict(origin="custom", timestamp_unit="ns", sources=[dict(path="raw", role="attack", assume_benign=True)]))
    with pytest.raises(ValueError, match="forbidden"):
        ingest(manifest, tmp_path)
