"""Real localhost HTTP: one server + three separate RSU processes, then evaluation.

Run from the project root after pip install -e ".[dev]":
    python scripts/network_smoke_test.py --out runs/network_smoke
Synthetic functionality check only; not a VeReMi detection result.
"""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import numpy as np
import torch
from fedcontext.config import DEFAULTS
from fedcontext.demo import make_demo
from fedcontext.prepare import prepare
from fedcontext.network.setup import partition
from fedcontext.network.finalize import finalize
from fedcontext.network.client import Transport
from fedcontext.utils import dump_json, fresh_dir, read_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = fresh_dir(args.out).resolve()
    cfg = dict(DEFAULTS, clients=3, partition='receiver_hash', rounds=2, local_epochs=1,
               calibration_min_windows=50, latency_samples=30, threads=1)
    manifest = make_demo(out/'raw')
    prepare(manifest, out/'prepared', cfg)
    shards = partition(out/'prepared', out/'shards', cfg)
    server_manifest = shards/'server/federation.json'
    members = read_json(server_manifest)['members']
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    url = f'http://127.0.0.1:{port}'
    prefix = [sys.executable, '-m', 'fedcontext.network']
    handles, clients = [], []
    server = None
    try:
        log = (out/'server.log').open('w'); handles.append(log)
        server = subprocess.Popen(prefix+['server','--manifest',str(server_manifest),
            '--out',str(out/'server_run'),'--port',str(port),'--round-timeout','180'], stdout=log, stderr=subprocess.STDOUT)
        # Readiness uses the server's public log; no repeated 45-second retries.
        deadline = time.monotonic()+30
        while 'Aggregator listening' not in (out/'server.log').read_text():
            if server.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError('Server failed to start; see server.log')
            time.sleep(.1)
        for client in sorted(members):
            log = (out/f'client_{client}.log').open('w'); handles.append(log)
            proc = subprocess.Popen(prefix+['client','--shard',str(shards/f'client_{client}'),
                '--server',url,'--out',str(out/f'client_{client}_run'),'--wait-timeout','180'], stdout=log, stderr=subprocess.STDOUT)
            clients.append(proc)
        for proc in clients:
            if proc.wait(timeout=180) != 0:
                raise RuntimeError('RSU failed; see client log')
        stats = read_json(out/'server_run/training.json')
        assert stats['finished'] and stats['completed_rounds'] == 2
        global_state = torch.load(out/'server_run/model.pt', map_location='cpu', weights_only=True)
        model_hash = stats['final_model_sha256']
        downloads = {}
        for client in sorted(members):
            info = read_json(out/f'client_{client}_run/download.json')
            downloads[client] = info['final_model_sha256']
            assert downloads[client] == model_hash
            local = torch.load(out/f'client_{client}_run/model.pt', map_location='cpu', weights_only=True)
            assert all(torch.equal(v, local[k]) for k, v in global_state.items())
        events = [json.loads(line) for line in (out/'server_run/network_events.jsonl').read_text().splitlines()]
        accepted = [e for e in events if e['event']=='update_accepted']
        assert len(accepted) == 2*len(members)
        assert len([e for e in events if e['event']=='round_aggregated']) == 2
        deployment = finalize(out/'prepared', out/'server_run', out/'deployable_model')
        from fedcontext.inference import StreamingDetector
        detector = StreamingDetector(deployment)
        for i in range(11):
            decision = detector.update(dict(rx_time=i*.1+.002, tx_time=i*.1,
                pseudonym='new_sender', message_id=str(i), x=i*2., y=0., rx_x=0., rx_y=0.,
                speed=20., acceleration=0., heading=90.))
        assert decision['reason'] == 'model_score'
        report = dict(origin='synthetic_demo', transport='HTTP over 127.0.0.1',
            separate_server_process=True, separate_client_processes=len(clients),
            clients={k:v['samples'] for k,v in members.items()}, completed_rounds=2,
            accepted_uploads=len(accepted), final_server_sha256=model_hash,
            client_final_sha256=downloads, exact_tensor_equality=True,
            calibration_and_test_evaluation_completed=True, streaming_inference_completed=True,
            actual_upload_body_bytes=sum(e['payload_bytes'] for e in accepted),
            model_download_body_bytes=sum(e['payload_bytes'] for e in events if e['event']=='model_download'),
            limitations='Synthetic functional test, not cloud deployment or VeReMi accuracy. Byte counts exclude HTTP/TLS/TCP overhead and status polling.')
        dump_json(out/'VERIFIED_NETWORK.json', report)
        print(json.dumps(report, indent=2))
    finally:
        for proc in clients:
            if proc.poll() is None:
                proc.terminate(); proc.wait(timeout=10)
        if server is not None and server.poll() is None:
            server.terminate(); server.wait(timeout=10)
        for handle in handles:
            handle.close()


if __name__ == '__main__':
    main()
