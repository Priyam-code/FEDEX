import numpy as np
import torch
from fedcontext.models import ThreeStream
from fedcontext.config import DEFAULTS
from fedcontext.features import ReceiverFeatures
from fedcontext.calibration import Calibrator
from fedcontext.inference import StreamingDetector
from fedcontext.utils import dump_json


def test_saved_model_streaming_matches_offline_feature_engine(tmp_path):
    torch.manual_seed(123)
    torch.set_num_threads(1)
    model = ThreeStream().eval()
    torch.save(model.state_dict(), tmp_path/"model.pt")
    cal = Calibrator().fit(np.arange(300).reshape(100, 3)/100, np.arange(300).reshape(100, 3)/101, .9, .99)
    dump_json(tmp_path/"calibration.json", cal.to_dict())
    dump_json(tmp_path/"run.json", dict(config=DEFAULTS, road_map=False))
    streaming = StreamingDetector(tmp_path)
    engine = ReceiverFeatures(DEFAULTS)
    for i in range(15):
        row = dict(rx_time=i*.1+.01, tx_time=i*.1, pseudonym="a", message_id=str(i),
                   x=i*2., y=0., rx_x=i*2.-10, rx_y=0., speed=20., acceleration=0., heading=90.)
        answer = streaming.update(row)
        features = engine.update(row)
        if features is None:
            assert answer["reason"] == "insufficient_history" and answer["score"] is None
        else:
            with torch.inference_mode():
                err = model.errors(*(torch.from_numpy(features[k][None]) for k in ("movement", "target", "timing", "context"))).numpy()
            scores, decisions, _ = cal.predict(err)
            assert answer["decision"] == decisions[0]
            assert answer["score"] == scores[0]
            np.testing.assert_allclose(answer["stream_mse"], err[0], atol=1e-7)


def test_current_movement_is_not_in_predictor_history():
    engine = ReceiverFeatures(DEFAULTS)
    for i in range(11):
        row = dict(rx_time=i*.1+.01, tx_time=i*.1, pseudonym="a", message_id=str(i),
                   x=i*2., y=0., rx_x=0., rx_y=0., speed=1000. if i==10 else 20., acceleration=0., heading=90.)
        f = engine.update(row)
    assert np.all(f["movement"][:, 2] == .5)
    assert f["target"][2] == 25.
