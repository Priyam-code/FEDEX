"""Calibrate a completed global checkpoint on a separate evaluation machine."""
from pathlib import Path
import shutil
import numpy as np
import torch
from ..models import ThreeStream
from ..prepare import load_prepared
from ..training import score
from ..calibration import Calibrator
from ..experiment import evaluate_saved
from ..utils import dump_json, fresh_dir, read_json
from .protocol import digest


def finalize(prepared, server_run, output):
    source = Path(server_run)
    run, stats = read_json(source/'run.json'), read_json(source/'training.json')
    if not stats['finished']:
        raise ValueError('Federated training has not completed')
    if digest((source/'global_weights.npz').read_bytes()) != stats['final_model_sha256']:
        raise ValueError('Global model payload hash mismatch')
    data, meta, info = load_prepared(prepared)
    cfg = run['config']
    if info['road_map'] or run['road_map']:
        raise ValueError('This network experiment requires no road map')
    for key in ('history', 'max_gap_s', 'neighbor_ttl_s', 'neighbor_radius_m'):
        if info['config'][key] != cfg[key]:
            raise ValueError(f'Feature configuration mismatch: {key}')
    groups = {}
    for name in ('score_ref', 'calibration'):
        part = meta[meta.partition == name]
        if len(part) < cfg['calibration_min_windows'] or part.label.ne(0).any() or not part.clean_window.all():
            raise ValueError(f'Need sufficient clean benign {name} windows')
        groups[name] = part.index.to_numpy()
    torch.set_num_threads(cfg['threads'])
    model = ThreeStream()
    model.load_state_dict(torch.load(source/'model.pt', map_location='cpu', weights_only=True))
    # Compare actual tensor values, not reserialization bytes.
    from .protocol import decode
    transmitted = decode((source/'global_weights.npz').read_bytes(), model.state_dict())
    if any(not torch.equal(v, transmitted[k]) for k, v in model.state_dict().items()):
        raise ValueError('Checkpoint does not match transmitted weights')
    ref = score(model, data, groups['score_ref'])
    cal = score(model, data, groups['calibration'])
    calibrator = Calibrator().fit(ref, cal, cfg['q_low'], cfg['q_high'])
    out = fresh_dir(output)
    shutil.copyfile(source/'model.pt', out/'model.pt')
    run['calibration_scope'] = 'Separate trusted offline evaluation machine; not federated calibration'
    dump_json(out/'run.json', run)
    dump_json(out/'calibration.json', calibrator.to_dict())
    shutil.copyfile(source/'training.json', out/'training.json')
    evaluate_saved(prepared, out, out/'evaluation')
    return out
