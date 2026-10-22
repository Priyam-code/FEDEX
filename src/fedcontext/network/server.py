"""Authenticated synchronous FedAvg coordinator, served over HTTP or HTTPS."""
import csv
import hmac
import json
import ssl
import threading
import time
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import torch
from ..models import ThreeStream
from ..training import fedavg
from ..utils import dump_json, fresh_dir, read_json, seed_everything
from .protocol import PROTOCOL, MAX_BODY, encode, decode, digest


class Conflict(ValueError):
    pass


class Coordinator:
    def __init__(self, manifest, output, round_timeout=600):
        self.info = read_json(manifest)
        if self.info['protocol'] != PROTOCOL or self.info['road_map']:
            raise ValueError('Unsupported protocol or road-map mode')
        self.cfg, self.members = self.info['config'], self.info['members']
        if len(self.members) < 2 or any(type(m['samples']) is not int or m['samples'] <= 0
                                      or not isinstance(m['token'], str) or len(m['token']) < 32
                                      for m in self.members.values()):
            raise ValueError('Need >=2 clients with positive sample counts and private tokens')
        if self.cfg['rounds'] < 1 or round_timeout <= 0:
            raise ValueError('Rounds and timeout must be positive')
        seed_everything(self.info['seed'], self.cfg['threads'])
        self.model = ThreeStream()
        self.body = encode(self.model.state_dict())
        self.sha = digest(self.body)
        self.out = fresh_dir(output)
        self.run_id = uuid.uuid4().hex
        self.round, self.pending, self.accepted = 1, {}, {}
        self.error = None
        self.started = time.monotonic()
        self.round_started, self.timeout = self.started, round_timeout
        self.lock = threading.RLock()
        self.round_rows = []
        self.downloads = {}
        self.round_uploads = {}
        dump_json(self.out/'run.json', dict(config=self.cfg, seed=self.info['seed'],
            origin=self.info['origin'], method='network_fedavg', road_map=False,
            federation_id=self.info['federation_id'], run_id=self.run_id,
            note='Weights only until separate benign calibration. Server has no training/selection/calibration data.'))
        self.event('server_started', participants=list(self.members))

    @property
    def finished(self):
        return self.round > self.cfg['rounds']

    def event(self, kind, **fields):
        with self.lock, (self.out/'network_events.jsonl').open('a', encoding='utf-8') as sink:
            sink.write(json.dumps(dict(time=datetime.now(timezone.utc).isoformat(),
                                      event=kind, **fields))+'\n')

    def check_timeout(self):
        if not self.finished and not self.error and time.monotonic()-self.round_started > self.timeout:
            self.error = 'Round deadline exceeded; missing clients: '+','.join(sorted(set(self.members)-set(self.pending)))
            dump_json(self.out/'failure.json', dict(round=self.round, error=self.error))
            self.event('failed', round=self.round, reason=self.error)

    def status(self, client):
        with self.lock:
            self.check_timeout()
            return dict(protocol=PROTOCOL, federation_id=self.info['federation_id'],
                run_id=self.run_id, round=self.round, rounds=self.cfg['rounds'],
                finished=self.finished, error=self.error, config=self.cfg,
                samples=self.members[client]['samples'], received=sorted(self.pending), model_sha256=self.sha)

    def snapshot(self, client):
        with self.lock:
            self.check_timeout()
            if self.error:
                raise Conflict(self.error)
            self.downloads[self.round] = self.downloads.get(self.round, 0)+len(self.body)
            self.event('model_download', client=client, round=self.round, payload_bytes=len(self.body), sha256=self.sha)
            return self.body, {'X-Round': str(self.round), 'X-Model-SHA256': self.sha,
                               'X-Run-ID': self.run_id}

    def submit(self, client, number, base_sha, run_id, body):
        with self.lock:
            self.check_timeout()
            if run_id != self.run_id:
                raise Conflict('Server run has changed; restart client with the correct server')
            if self.error:
                raise Conflict(self.error)
            key, upload_sha = (number, client), digest(body)
            if key in self.accepted:
                old_base, old_sha = self.accepted[key]
                if (base_sha, upload_sha) != (old_base, old_sha):
                    raise Conflict('A different update was already accepted for this client/round')
                self.event('duplicate_upload', client=client, round=number, payload_bytes=len(body))
                return dict(accepted=True, duplicate=True, round=number)
            if self.finished or number != self.round or base_sha != self.sha:
                raise Conflict('Stale round or wrong base-model hash')
            state = decode(body, self.model.state_dict())
            self.pending[client] = state
            self.accepted[key] = (base_sha, upload_sha)
            self.round_uploads[number] = self.round_uploads.get(number, 0)+len(body)
            self.event('update_accepted', client=client, round=number, payload_bytes=len(body),
                       samples=self.members[client]['samples'], sha256=upload_sha)
            if len(self.pending) == len(self.members):
                try:
                    order = sorted(self.members)
                    combined = fedavg([self.pending[c] for c in order], [self.members[c]['samples'] for c in order])
                    self.model.load_state_dict(combined)
                    self.body = encode(combined)
                    self.sha = digest(self.body)
                    # Save every completed global model atomically.
                    temp = self.out/'model.partial'
                    torch.save(combined, temp)
                    temp.replace(self.out/'model.pt')
                    (self.out/'global_weights.npz').write_bytes(self.body)
                    self.round_rows.append(dict(round=number, participants=len(order),
                        train_windows=sum(m['samples'] for m in self.members.values()),
                        unique_upload_body_bytes=self.round_uploads[number],
                        model_download_body_bytes=self.downloads.get(number, 0),
                        round_seconds=time.monotonic()-self.round_started, global_sha256=self.sha))
                    with (self.out/'communication.csv').open('w', newline='', encoding='utf-8') as sink:
                        writer = csv.DictWriter(sink, fieldnames=list(self.round_rows[0]))
                        writer.writeheader(); writer.writerows(self.round_rows)
                    self.event('round_aggregated', round=number, sha256=self.sha)
                    self.pending = {}
                    self.round += 1
                    self.round_started = time.monotonic()
                    dump_json(self.out/'training.json', dict(mode='network_fedavg',
                        completed_rounds=number, planned_rounds=self.cfg['rounds'],
                        active_clients=len(order), seconds=time.monotonic()-self.started,
                        final_model_sha256=self.sha, finished=self.finished,
                        communication_scope='Recorded HTTP model body bytes; excludes HTTP/TLS/TCP headers and status polling. Final-model downloads are in network_events.jsonl.',
                        privacy='Per-client bearer authentication; optional TLS. Server sees individual weights. No secure aggregation, DP, poisoning defence or server restart recovery.'))
                    print(f'FedAvg round {number}/{self.cfg["rounds"]} complete; model {self.sha[:12]}', flush=True)
                except Exception:
                    self.error = 'Aggregation/checkpoint failed; restart a fresh run'
                    dump_json(self.out/'failure.json', dict(round=number, error=self.error))
                    raise
            return dict(accepted=True, duplicate=False, round=number)


def make_server(coordinator, host='127.0.0.1', port=8080, cert=None, key=None, allow_insecure_http=False):
    if bool(cert) != bool(key):
        raise ValueError('Supply both TLS certificate and key')
    if host not in ('127.0.0.1', 'localhost', '::1') and not cert and not allow_insecure_http:
        raise ValueError('Remote binding requires TLS, or --allow-insecure-http on a trusted test LAN')

    class Handler(BaseHTTPRequestHandler):
        server_version = 'FedContextResearch/1'

        def log_message(self, *args):
            pass  # Never log bearer tokens.

        def reply(self, code, value, headers=None):
            binary = isinstance(value, bytes)
            body = value if binary else json.dumps(value).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/octet-stream' if binary else 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            for name, val in (headers or {}).items():
                self.send_header(name, val)
            self.end_headers()
            self.wfile.write(body)

        def authenticate(self):
            client = self.headers.get('X-Client-ID', '')
            member = coordinator.members.get(client)
            token = self.headers.get('Authorization', '')
            if not member or not hmac.compare_digest(token, 'Bearer '+member['token']):
                self.reply(401, dict(error='Authentication failed'))
                return None
            return client

        def do_GET(self):
            client = self.authenticate()
            if client is None:
                return
            if self.path == '/status':
                self.reply(200, coordinator.status(client))
            elif self.path == '/model':
                try:
                    body, headers = coordinator.snapshot(client)
                    self.reply(200, body, headers)
                except Conflict as exc:
                    self.reply(409, dict(error=str(exc)))
            else:
                self.reply(404, dict(error='Unknown endpoint'))

        def do_POST(self):
            client = self.authenticate()
            if client is None:
                return
            if self.path != '/update':
                self.reply(404, dict(error='Unknown endpoint')); return
            try:
                if self.headers.get('Transfer-Encoding'):
                    raise ValueError('Chunked uploads are not supported')
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= MAX_BODY:
                    self.reply(413, dict(error='Invalid model body size')); return
                self.connection.settimeout(30)
                body = self.rfile.read(length)
                if len(body) != length:
                    raise ValueError('Incomplete upload')
                answer = coordinator.submit(client, int(self.headers['X-Round']),
                    self.headers.get('X-Base-SHA256', ''), self.headers.get('X-Run-ID', ''), body)
                self.reply(200, answer)
            except Conflict as exc:
                self.reply(409, dict(error=str(exc)))
            except (ValueError, TypeError, KeyError, OSError) as exc:
                self.reply(400, dict(error=str(exc)))
            except Exception:
                self.reply(500, dict(error='Server failed to complete this update; inspect server logs'))

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    if cert:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.load_cert_chain(cert, key)
        server.socket = context.wrap_socket(server.socket, server_side=True)
    return server
