"""An RSU process with access to only its own local training shard."""
import csv
import json
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
import numpy as np
import torch
from ..models import ThreeStream
from ..training import local_train
from ..utils import dump_json, fresh_dir, read_json, seed_everything
from .protocol import PROTOCOL, MAX_BODY, encode, decode, digest


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None  # Never forward credentials to another endpoint.


class Transport:
    def __init__(self, url, info, ca_file=None, allow_insecure_http=False):
        target = urllib.parse.urlsplit(url)
        if target.scheme not in ('http', 'https') or not target.hostname or target.username or target.password or target.query or target.fragment or target.path not in ('', '/'):
            raise ValueError('Server URL must be http(s)://host:port without credentials or a path')
        if target.scheme == 'http' and target.hostname not in ('localhost', '127.0.0.1', '::1') and not allow_insecure_http:
            raise ValueError('Remote clients require HTTPS or explicit --allow-insecure-http for a trusted test LAN')
        self.url, self.info = url.rstrip('/'), info
        context = ssl.create_default_context(cafile=ca_file)
        self.opener = urllib.request.build_opener(NoRedirect(), urllib.request.HTTPSHandler(context=context))

    def request(self, path, body=None, extra=None):
        headers = {'Authorization': 'Bearer '+self.info['token'], 'X-Client-ID': self.info['client_id']}
        headers.update(extra or {})
        if body is not None:
            headers['Content-Type'] = 'application/octet-stream'
        request = urllib.request.Request(self.url+path, data=body, headers=headers,
                                         method='GET' if body is None else 'POST')
        for attempt in range(5):
            try:
                with self.opener.open(request, timeout=45) as response:
                    payload = response.read(MAX_BODY+1)
                    if len(payload) > MAX_BODY:
                        raise ValueError('Server response is too large')
                    return payload, response.headers
            except urllib.error.HTTPError as exc:
                detail = exc.read(2048).decode(errors='replace')
                raise ValueError(f'Server HTTP {exc.code}: {detail}') from None
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
                if attempt == 4:
                    raise
                time.sleep(min(2**attempt, 8))


def load_local(shard):
    shard = Path(shard)
    info = read_json(shard/'client.json')
    if info['protocol'] != PROTOCOL or info['road_map']:
        raise ValueError('Unsupported client shard')
    with np.load(shard/'train.npz', allow_pickle=False) as source:
        data = {k: source[k] for k in source.files}
    n, h = info['samples'], info['config']['history']
    shapes = dict(movement=(n, h, 6), target=(n, 6), timing=(n, h, 7), context=(n, 10))
    if n <= 0 or set(data) != set(shapes):
        raise ValueError('Invalid local training arrays')
    for k, shape in shapes.items():
        if data[k].shape != shape or data[k].dtype != np.float32 or not np.isfinite(data[k]).all():
            raise ValueError(f'Invalid local array: {k}')
    return info, data


def run_client(shard, url, output, ca_file=None, allow_insecure_http=False, wait_timeout=1800):
    if wait_timeout <= 0:
        raise ValueError('Wait timeout must be positive')
    info, data = load_local(shard)
    cfg = info['config']
    seed_everything(info['seed'], cfg['threads'])
    transport = Transport(url, info, ca_file, allow_insecure_http)
    out = fresh_dir(output)
    history, run_id, last_progress = [], None, time.monotonic()
    last_round = None
    while True:
        payload, _ = transport.request('/status')
        status = json.loads(payload)
        if status['protocol'] != PROTOCOL or status['federation_id'] != info['federation_id'] or status['config'] != cfg or status['samples'] != info['samples']:
            raise ValueError('Client/server experiment settings do not match')
        if run_id is not None and run_id != status['run_id']:
            raise ValueError('Server restarted; use a fresh coordinated run')
        run_id = status['run_id']
        if status['error']:
            raise ValueError(status['error'])
        if status['round'] != last_round:
            last_round, last_progress = status['round'], time.monotonic()
        if time.monotonic()-last_progress > wait_timeout:
            raise TimeoutError('Timed out waiting for all RSUs to complete a round')
        if not status['finished'] and info['client_id'] in status['received']:
            time.sleep(.5)
            continue
        body, headers = transport.request('/model')
        number, sha = int(headers['X-Round']), digest(body)
        if headers['X-Run-ID'] != run_id or headers['X-Model-SHA256'] != sha:
            raise ValueError('Server model identity/hash mismatch')
        if number != status['round'] or sha != status['model_sha256']:
            continue  # Round advanced between status and model download.
        model = ThreeStream()
        model.load_state_dict(decode(body, model.state_dict()))
        if status['finished']:
            torch.save(model.state_dict(), out/'model.pt')
            (out/'global_weights.npz').write_bytes(body)
            dump_json(out/'download.json', dict(client_id=info['client_id'], run_id=run_id,
                completed_rounds=number-1, final_model_sha256=sha, payload_bytes=len(body)))
            print(f'RSU {info["client_id"]}: final global model downloaded ({sha[:12]})', flush=True)
            return out
        seed_everything(info['seed']+number*1000+int(info['client_id']), cfg['threads'])
        started = time.perf_counter()
        losses = local_train(model, data, np.arange(info['samples']), cfg, cfg['local_epochs'])
        update = encode(model.state_dict())
        answer, _ = transport.request('/update', update, {'X-Round': str(number),
            'X-Base-SHA256': sha, 'X-Run-ID': run_id})
        if not json.loads(answer)['accepted']:
            raise ValueError('Server did not accept update')
        history.append(dict(round=number, samples=info['samples'], base_sha256=sha,
            update_sha256=digest(update), download_body_bytes=len(body), upload_body_bytes=len(update),
            local_train_and_upload_seconds=time.perf_counter()-started,
            movement_loss=float(losses[0]), timing_loss=float(losses[1]), context_loss=float(losses[2])))
        with (out/'client_rounds.csv').open('w', newline='', encoding='utf-8') as sink:
            writer = csv.DictWriter(sink, fieldnames=list(history[0]))
            writer.writeheader(); writer.writerows(history)
        last_progress = time.monotonic()
        print(f'RSU {info["client_id"]}: round {number} trained and uploaded', flush=True)
