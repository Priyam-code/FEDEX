"""python -m fedcontext.network {partition,server,client,finalize}."""
import argparse
from .setup import partition
from .server import Coordinator, make_server
from .client import run_client
from .finalize import finalize
from ..config import load_config


def main():
    parser = argparse.ArgumentParser(description='Networked FedAvg research prototype')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('partition', help='Prepare offline local client shards and server metadata')
    p.add_argument('--prepared', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--seed', type=int, default=7)
    s = sub.add_parser('server', help='Run the aggregation service; Ctrl+C after clients download final model')
    s.add_argument('--manifest', required=True)
    s.add_argument('--out', required=True)
    s.add_argument('--host', default='127.0.0.1')
    s.add_argument('--port', type=int, default=8080)
    s.add_argument('--round-timeout', type=float, default=600)
    s.add_argument('--cert')
    s.add_argument('--key')
    s.add_argument('--allow-insecure-http', action='store_true')
    c = sub.add_parser('client', help='Run one RSU process on its own shard')
    c.add_argument('--shard', required=True)
    c.add_argument('--server', required=True)
    c.add_argument('--out', required=True)
    c.add_argument('--ca-file')
    c.add_argument('--allow-insecure-http', action='store_true')
    c.add_argument('--wait-timeout', type=float, default=1800)
    f = sub.add_parser('finalize', help='Offline calibration and evaluation; no retraining')
    f.add_argument('--prepared', required=True)
    f.add_argument('--server-run', required=True)
    f.add_argument('--out', required=True)
    args = parser.parse_args()
    try:
        if args.command == 'partition':
            print(partition(args.prepared, args.out, load_config(args.config), args.seed))
        elif args.command == 'server':
            coordinator = Coordinator(args.manifest, args.out, args.round_timeout)
            server = make_server(coordinator, args.host, args.port, args.cert, args.key, args.allow_insecure_http)
            print(f'Aggregator listening on {args.host}:{server.server_port}; waiting for {len(coordinator.members)} RSUs', flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
            finally:
                server.server_close()
        elif args.command == 'client':
            print(run_client(args.shard, args.server, args.out, args.ca_file,
                             args.allow_insecure_http, args.wait_timeout))
        else:
            print(finalize(args.prepared, args.server_run, args.out))
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(2, f'ERROR: {exc}\n')


if __name__ == '__main__':
    main()
