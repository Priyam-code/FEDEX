"""One-time offline benchmark partitioning, before distributing client folders."""
import secrets
import uuid
import numpy as np
from ..prepare import load_prepared
from ..utils import dump_json, fresh_dir
from .protocol import PROTOCOL


def partition(prepared, output, config, seed=7):
    data, meta, info = load_prepared(prepared)
    for key in ('history', 'clients', 'partition', 'region_width_m', 'max_gap_s',
                'neighbor_ttl_s', 'neighbor_radius_m'):
        if info['config'][key] != config[key]:
            raise ValueError(f'Prepared config mismatch: {key}')
    if info['road_map']:
        raise ValueError('Network version currently supports neighbour context without a road map')
    train = meta[meta.partition == 'train']
    if train.empty or train.label.ne(0).any() or not train.clean_window.all():
        raise ValueError('Need clean benign training windows')
    clients = {str(int(c)): g.index.to_numpy() for c, g in train.groupby('client')}
    if len(clients) < 2:
        raise ValueError('Need at least two nonempty clients')
    root = fresh_dir(output)
    common = dict(protocol=PROTOCOL, federation_id=uuid.uuid4().hex, config=config,
                  seed=seed, origin=info['origin'], road_map=False)
    members = {}
    for client, ids in clients.items():
        folder = fresh_dir(root/f'client_{client}')
        token = secrets.token_urlsafe(32)
        members[client] = dict(samples=len(ids), token=token)
        np.savez_compressed(folder/'train.npz', **{k: v[ids] for k, v in data.items()})
        dump_json(folder/'client.json', dict(**common, client_id=client, samples=len(ids), token=token))
        (folder/'client.json').chmod(0o600)
    server = fresh_dir(root/'server')
    dump_json(server/'federation.json', dict(**common, members=members))
    (server/'federation.json').chmod(0o600)
    dump_json(root/'distribution.json', dict(clients={k: len(v) for k, v in clients.items()},
        note='Distribute only client_N to RSU N. Server receives only server/federation.json, never train.npz. Keep tokens private.'))
    return root
