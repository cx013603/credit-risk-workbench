"""E14 loopback JSON transport over one verified E10 snapshot; no model logic."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import socket
import threading

from .raw import DataError, sha256_file
from .scoring import load_scoring_bundle, validate_ids

VERSION = 'scoring_api_v1'
POLICY = {'max_batch_ids': 100, 'max_body_bytes': 8192, 'max_connections': 2,
          'socket_timeout_seconds': 3}
CODE = ['src/credit_risk/scoring_api.py', 'scripts/run_scoring_api.py']


def load_api_config(root, path):
    root = Path(root).resolve()
    try:
        v = json.loads(Path(path).read_text(encoding='utf-8-sig'))
        if (set(v) != {'version', 'scoring_report', 'host', 'port', *POLICY}
                or v['version'] != VERSION or v['host'] != '127.0.0.1'
                or type(v['port']) is not int or not 0 <= v['port'] <= 65535
                or any(type(v[k]) is not int or v[k] != n for k, n in POLICY.items())):
            raise ValueError('unsupported policy')
        report = Path(v['scoring_report'])
        if (report.is_absolute() or report.name != 'run.json'
                or not (root/report).resolve().is_relative_to(root/'reports/runs')):
            raise ValueError('project report required')
        return {**v, 'report_path': (root/report).resolve()}
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise DataError('api_config', 'Exact local API configuration required') from exc


def _object(pairs):
    result = {}
    for k, v in pairs:
        if k in result:
            raise ValueError('duplicate JSON key')
        result[k] = v
    return result


def parse_request(body, batch):
    try:
        v = json.loads(body.decode('utf-8'), object_pairs_hook=_object,
                       parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
        key = 'ids' if batch else 'id'
        if not isinstance(v, dict) or set(v) != {key}:
            raise ValueError('exact ID schema required')
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise DataError('api_json', 'UTF-8 JSON object with exactly id or ids required') from exc
    ids = v[key] if batch else [v[key]]
    if not isinstance(ids, list) or len(ids) > POLICY['max_batch_ids']:
        raise DataError('api_batch_limit', 'Batch must contain 1 to 100 unique integer IDs')
    return list(validate_ids(ids))


class ApiState:
    def __init__(self, root, report):
        self.root, self.report_path = Path(root).resolve(), Path(report).resolve()
        self.session = None
        self.phase = 'created'
        self.error_code = None
        self.load_attempts = 0
        self.score_lock = threading.Lock()

    def load(self):
        if self.phase != 'created':
            raise DataError('api_load_once', 'Start a new process for a different immutable snapshot')
        self.phase = 'loading'
        self.load_attempts += 1
        try:
            self.session = load_scoring_bundle(self.root, self.report_path)
            self.report_sha256 = sha256_file(self.report_path)
            self.phase = 'ready'
        except Exception as exc:
            self.error_code = exc.code if isinstance(exc, DataError) else type(exc).__name__
            self.phase = 'failed'
            raise
        return self

    def health(self):
        return {'api_version': VERSION, 'status': self.phase, 'ready': self.phase == 'ready',
                'error_code': self.error_code, 'model_load_attempts': self.load_attempts,
                'scope': 'existing_labeled_applications_replay',
                'scoring_report_sha256': getattr(self, 'report_sha256', None), **POLICY}

    def score(self, ids):
        if self.phase != 'ready':
            raise DataError('api_unavailable', 'Verified scoring snapshot unavailable')
        if not self.score_lock.acquire(blocking=False):
            raise DataError('api_busy', 'Scoring is busy; retry this bounded request later')
        try:
            if self.phase != 'ready':
                raise DataError('api_unavailable', 'Verified scoring snapshot unavailable')
            scores, components = self.session.score_ids(ids)
            return {'api_version': VERSION, 'scope': 'existing_labeled_applications_replay',
                    'scoring_report_sha256': self.report_sha256, 'count': len(ids),
                    'results': scores.to_pylist(), 'scorecard_components': components.to_pylist(),
                    'meaning': 'risk_flag = tree probability >= original validation threshold; not a lending decision. Components explain only the scorecard, not tree probability.',
                    'sample_notice': 'train is workflow demonstration; validation was used for selection; test is application-level holdout with disclosed early exposure.',
                    'limitations': self.session.bundle['limitations']}
        except DataError as exc:
            if exc.code not in {'scoring_unknown_id', 'scoring_ids', 'scoring_id_type',
                                'scoring_duplicate_id', 'api_unavailable'}:
                self.phase, self.error_code = 'failed', exc.code
            raise
        except Exception as exc:
            self.phase, self.error_code = 'failed', type(exc).__name__
            raise DataError('api_unavailable', 'Scoring failed; restart and verify package before reuse') from exc
        finally:
            self.score_lock.release()


def error_payload(code, message):
    return {'api_version': VERSION, 'error': {'code': code, 'message': message}}


class BoundedServer(ThreadingHTTPServer):
    # At most two connection threads; only one may enter the E10 session.
    # No executor queue and no background worker/model copies.
    daemon_threads = False
    block_on_close = True
    request_queue_size = 2
    allow_reuse_address = False

    def __init__(self, address, handler, state):
        self.state = state
        self.slots = threading.BoundedSemaphore(POLICY['max_connections'])
        super().__init__(address, handler)

    def process_request(self, request, client_address):
        request.settimeout(POLICY['socket_timeout_seconds'])
        if not self.slots.acquire(blocking=False):
            body = json.dumps(error_payload('api_busy', 'Connection capacity reached; retry later')).encode()
            try:
                request.sendall(b'HTTP/1.0 429 Too Many Requests\r\nContent-Type: application/json\r\nConnection: close\r\nRetry-After: 1\r\nContent-Length: '
                                + str(len(body)).encode() + b'\r\n\r\n' + body)
            except OSError:
                pass
            finally:
                self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()


class Handler(BaseHTTPRequestHandler):
    server_version = 'CreditRiskLocalAPI/1'
    sys_version = ''

    def log_message(self, *_):
        pass

    def send_error(self, code, message=None, explain=None):
        # Also keep parser-level and unknown-method failures in the JSON contract.
        self.fail(code, 'api_http', 'HTTP request rejected')

    def reply(self, status, value):
        body = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Connection', 'close')
        if status == 429:
            self.send_header('Retry-After', '1')
        if status == 405:
            self.send_header('Allow', 'GET' if self.path.startswith('/health/') else 'POST')
        self.end_headers()
        self.close_connection = True
        try:
            if self.command != 'HEAD':
                self.wfile.write(body)
        except (OSError, socket.timeout):
            pass

    def fail(self, status, code, message):
        self.reply(status, error_payload(code, message))

    def do_GET(self):
        if self.path in {'/health/live', '/health/ready'}:
            status = 200 if self.path == '/health/live' or self.server.state.phase == 'ready' else 503
            self.reply(status, self.server.state.health())
        elif self.path in {'/v1/score', '/v1/score/batch'}:
            self.fail(405, 'api_method', 'POST required')
        else:
            self.fail(404, 'api_route', 'Unknown endpoint; query parameters are unsupported')

    def do_POST(self):
        if self.path in {'/health/live', '/health/ready'}:
            self.fail(405, 'api_method', 'GET required')
            return
        if self.path not in {'/v1/score', '/v1/score/batch'}:
            self.fail(404, 'api_route', 'Unknown endpoint; query parameters are unsupported')
            return
        lengths = self.headers.get_all('Content-Length', [])
        if self.headers.get('Transfer-Encoding') or self.headers.get('Content-Encoding'):
            self.fail(400, 'api_framing', 'Encoded or chunked bodies are unsupported')
            return
        if not lengths:
            self.fail(411, 'api_length', 'Content-Length required')
            return
        if len(lengths) != 1 or not lengths[0].isascii() or not lengths[0].isdigit():
            self.fail(400, 'api_length', 'One nonnegative decimal Content-Length required')
            return
        if len(lengths[0]) > 10:
            self.fail(413, 'api_body_limit', 'JSON body exceeds 8192 bytes')
            return
        length = int(lengths[0])
        if length > POLICY['max_body_bytes']:
            self.fail(413, 'api_body_limit', 'JSON body exceeds 8192 bytes')
            return
        types = self.headers.get_all('Content-Type', [])
        if len(types) != 1 or types[0].lower().replace(' ', '') not in {'application/json', 'application/json;charset=utf-8'}:
            self.fail(415, 'api_media_type', 'application/json UTF-8 required')
            return
        try:
            body = self.rfile.read(length)
            if len(body) != length:
                self.fail(400, 'api_body', 'Incomplete JSON body')
                return
            ids = parse_request(body, self.path.endswith('/batch'))
            self.reply(200, self.server.state.score(ids))
        except socket.timeout:
            self.fail(408, 'api_timeout', 'Request body read timed out')
        except DataError as exc:
            status = (429 if exc.code == 'api_busy' else 503 if self.server.state.phase != 'ready'
                      else 404 if exc.code == 'scoring_unknown_id' else 400)
            self.fail(status, exc.code, str(exc))
        except (OSError, ValueError, RecursionError):
            self.fail(400, 'api_body', 'Invalid request body')

    def do_other(self):
        self.fail(405, 'api_method', 'Supported methods are GET health and POST scoring')

    do_PUT = do_DELETE = do_PATCH = do_OPTIONS = do_HEAD = do_other


def make_server(state, host='127.0.0.1', port=8766):
    if host != '127.0.0.1' or type(port) is not int or not 0 <= port <= 65535:
        raise DataError('api_bind', 'Only IPv4 loopback is supported')
    if state.phase != 'ready':
        raise DataError('api_unavailable', 'Verify the complete E10 package before binding')
    return BoundedServer((host, port), Handler, state)
