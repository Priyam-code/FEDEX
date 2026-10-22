"""Protocol safety and coordinator correctness; real subprocess exchange has a separate smoke script."""
import copy
import io
import json
import threading
import urllib.error
import urllib.request
import numpy as np
import pytest
import torch
from fedcontext.config import DEFAULTS
from fedcontext.models import ThreeStream
from fedcontext.network.protocol import encode, decode
from fedcontext.network.server import Coordinator, Conflict, make_server
from fedcontext.network.client import Transport
from fedcontext.utils import dump_json


def coordinator(tmp_path, rounds=2):
    path = tmp_path/'federation.json'
    info = dict(protocol=1, federation_id='test', seed=7, origin='synthetic_demo', road_map=False,
                config=dict(DEFAULTS, rounds=rounds), members={
                    '0': dict(samples=10, token='a'*40), '1': dict(samples=30, token='b'*40)})
    dump_json(path, info)
    return Coordinator(path, tmp_path/'run'), info


def test_numeric_model_roundtrip_and_rejection():
    state = ThreeStream().state_dict()
    restored = decode(encode(state))
    assert all(torch.equal(state[k], restored[k]) for k in state)
    bad = copy.deepcopy(state)
    bad[next(iter(bad))].flatten()[0] = float('nan')
    with pytest.raises(ValueError, match='Non-finite'):
        decode(encode(bad))
    with pytest.raises(ValueError):
        decode(b'not an archive')
    bad = copy.deepcopy(state)
    bad.pop(next(iter(bad)))
    with pytest.raises(ValueError, match='parameters'):
        decode(encode(bad))
    bad = copy.deepcopy(state)
    bad[next(iter(bad))] = torch.zeros(1)
    with pytest.raises(ValueError, match='shape/dtype'):
        decode(encode(bad))


def test_weighted_fedavg_and_round_rejection(tmp_path):
    c, _ = coordinator(tmp_path)
    state = c.model.state_dict()
    first = encode({k: torch.ones_like(v)*2 for k, v in state.items()})
    second = encode({k: torch.ones_like(v)*6 for k, v in state.items()})
    base, run = c.sha, c.run_id
    c.submit('0', 1, base, run, first)
    assert c.round == 1
    assert c.submit('0', 1, base, run, first)['duplicate']
    with pytest.raises(Conflict):
        c.submit('0', 1, base, run, second)
    c.submit('1', 1, base, run, second)
    assert c.round == 2
    assert all(torch.equal(v, torch.full_like(v, 5)) for v in c.model.state_dict().values())
    assert c.submit('1', 1, base, run, second)['duplicate']  # Lost response is retry-safe.
    with pytest.raises(Conflict):
        c.submit('0', 2, base, run, first)
    with pytest.raises(Conflict):
        c.submit('0', 2, c.sha, 'wrong-run', first)
    c.submit('0', 2, c.sha, run, first)
    c.submit('1', 2, c.sha, run, second)
    assert c.finished
    assert json.loads((c.out/'training.json').read_text())['finished']


def test_authentication_and_real_http_body(tmp_path):
    c, info = coordinator(tmp_path)
    server = make_server(c, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{server.server_port}'
    try:
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(url+'/status')
        assert exc.value.code == 401
        transport = Transport(url, dict(client_id='0', token=info['members']['0']['token']))
        body, headers = transport.request('/model')
        assert headers['X-Model-SHA256'] == c.sha
        assert body == c.body
        with pytest.raises(ValueError, match='HTTP 409'):
            transport.request('/update', body, {'X-Round': '99', 'X-Run-ID':c.run_id, 'X-Base-SHA256':c.sha})
        answer, _ = transport.request('/update', body, {'X-Round':'1', 'X-Run-ID':c.run_id, 'X-Base-SHA256':c.sha})
        assert json.loads(answer)['accepted']
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=5)


def test_remote_http_requires_explicit_opt_in(tmp_path):
    c, _ = coordinator(tmp_path)
    with pytest.raises(ValueError, match='TLS'):
        make_server(c, '0.0.0.0')
    with pytest.raises(ValueError, match='HTTPS'):
        Transport('http://192.0.2.1:8080', {})


def test_round_timeout_keeps_incomplete_run_unfinished(tmp_path):
    c, _ = coordinator(tmp_path)
    c.round_started -= 1000
    assert c.status('0')['error']
    assert not c.finished
    assert not (c.out/'model.pt').exists()


def test_https_with_verified_local_certificate(tmp_path):
    import shutil
    import subprocess
    if not shutil.which('openssl'):
        pytest.skip('OpenSSL command is needed only to generate this test certificate')
    cert, key = tmp_path/'cert.pem', tmp_path/'key.pem'
    subprocess.run(['openssl','req','-x509','-newkey','rsa:2048','-nodes','-days','1',
        '-keyout',str(key),'-out',str(cert),'-subj','/CN=localhost',
        '-addext','subjectAltName=DNS:localhost,IP:127.0.0.1'], check=True, capture_output=True)
    c, info = coordinator(tmp_path)
    server = make_server(c, port=0, cert=str(cert), key=str(key))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        transport = Transport(f'https://localhost:{server.server_port}',
            dict(client_id='0', token=info['members']['0']['token']), ca_file=str(cert))
        body, _ = transport.request('/model')
        assert body == c.body
        answer, _ = transport.request('/update', body, {'X-Round':'1', 'X-Run-ID':c.run_id, 'X-Base-SHA256':c.sha})
        assert json.loads(answer)['accepted']
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=5)
