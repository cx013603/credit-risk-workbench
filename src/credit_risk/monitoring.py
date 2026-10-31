"""E15 bounded response telemetry and train-only descriptive distributions.

No fitting, labels, model-quality claims or edits to accepted E10/E14 sources.
"""
from collections import Counter
from http.server import ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import socket
import threading
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .raw import DataError, sha256_file
from .runtime import now
from .scoring import load_scoring_bundle
from .scoring_api import ApiState, BoundedServer, Handler, POLICY as API_POLICY, error_payload

VERSION = 'monitoring_v1'
CODE = ['src/credit_risk/monitoring.py', 'scripts/run_monitoring.py']
POLICY = {'reference_partition': 'train', 'numeric_quantiles': 10, 'minimum_observations': 100,
          'tvd_review_threshold': .1, 'missing_rate_review_threshold': .05,
          'max_events': 1000, 'max_log_bytes': 8388608}
NOTICE = ['Existing-application replay; occurrence-weighted, not distinct people or a natural live cohort.',
          'Repeated IDs and selected examples change the observed distribution; no new labels or model performance inference.',
          'TVD/missing-rate thresholds are fixed classroom review heuristics, not significance tests or lending policy.',
          'Counts are responses prepared by the server, not confirmation of client receipt.',
          'Capacity rejection precedes HTTP parsing and has null latency; other latency excludes network response delivery.',
          'Health and summary reads excluded; incomplete headers/connection resets before a response are not counted.',
          'Original early exposure of 769 test IDs, person/time/target X/Y unknown, limited 40 columns and unverified licence remain.']


def save_json(path, value, exclusive=False):
    path = Path(path)
    if exclusive:
        with path.open('x', encoding='utf-8') as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write('\n')
    else:
        if path.exists():
            raise DataError('monitor_output_exists','Refuse to overwrite a published monitor output')
        partial = path.with_suffix(path.suffix+'.partial')
        with partial.open('x',encoding='utf-8') as f:
            json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        partial.replace(path)


def load_monitor_config(root, path):
    root = Path(root).resolve()
    try:
        v = json.loads(Path(path).read_text(encoding='utf-8-sig'))
        if (set(v) != {'version', 'scoring_report', 'host', 'port', *POLICY}
                or v['version'] != VERSION or v['host'] != '127.0.0.1'
                or type(v['port']) is not int or not 0 <= v['port'] <= 65535
                or any(type(v[k]) is not type(x) or v[k] != x for k, x in POLICY.items())):
            raise ValueError('unsupported local policy')
        r = Path(v['scoring_report'])
        if r.is_absolute() or r.name != 'run.json' or not (root/r).resolve().is_relative_to(root/'reports/runs'):
            raise ValueError('project scoring report required')
        return {**v, 'report_path': (root/r).resolve()}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise DataError('monitor_config', 'Exact local monitoring policy required') from exc


def histogram(spec, values):
    counts = [0]*spec['buckets']
    for value in values:
        if value is None:
            counts[0] += 1
            continue
        if spec['value_type'] == 'numeric':
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise DataError('monitor_value', 'Finite numeric value or genuine null required')
        elif not isinstance(value, str):
            raise DataError('monitor_value', 'String category or genuine null required')
        kind = spec['kind']
        if kind == 'all_missing':
            index = 1
        elif kind == 'categorical':
            index = spec['categories'].index(value)+1 if value in spec['categories'] else spec['buckets']-1
        elif kind == 'constant':
            c = spec['constant']
            index = 1 if value < c else 2 if value == c else 3
        else:
            index = int(np.searchsorted(spec['cuts'], value, side='right'))+1
        counts[index] += 1
    return counts


def make_spec(values, numeric):
    nonnull = [v for v in values if v is not None]
    spec = {'value_type': 'numeric' if numeric else 'categorical'}
    if not nonnull:
        spec.update(kind='all_missing', buckets=2)
    elif not numeric:
        categories = sorted(set(nonnull))
        spec.update(kind='categorical', categories=categories, buckets=len(categories)+2)
    elif min(nonnull) == max(nonnull):
        spec.update(kind='constant', constant=float(nonnull[0]), buckets=4)
    else:
        cuts = sorted(set(float(x) for x in np.quantile(nonnull, np.arange(1,10)/10, method='linear')))
        spec.update(kind='numeric', cuts=cuts, buckets=len(cuts)+2)
    spec['reference_counts'] = histogram(spec, values)
    return spec


def build_reference(root, config, record):
    root = Path(root).resolve()
    directory = root/'reports/monitoring/reference'/record.run_id
    directory.parent.mkdir(parents=True, exist_ok=True)
    if directory.exists():
        raise DataError('monitor_exists', 'Reference run cannot overwrite an existing directory')
    session = load_scoring_bundle(root, config['report_path'])
    training = session.features.filter(pc.equal(session.features['sample_use'], 'train'))
    columns = session.contract['feature_columns']
    fields = {}
    for name in columns:
        dtype = training.schema.field(name).type
        numeric = pa.types.is_integer(dtype) or pa.types.is_floating(dtype)
        fields[name] = make_spec(training[name].to_pylist(), numeric)
    # Saved E10 predictions only, never read the original test feature/label table.
    scores = pq.read_table(session.directory/'scores.parquet', columns=['SK_ID_CURR','sample_use','risk_probability'],
                           filters=[('sample_use', '=', 'train')])
    if (scores.num_rows != training.num_rows or set(scores['SK_ID_CURR'].to_pylist()) != set(training['SK_ID_CURR'].to_pylist())
            or set(scores['sample_use'].to_pylist()) != {'train'}):
        raise DataError('monitor_reference_population', 'Training-only reference IDs must agree')
    reference = {'version': VERSION, 'module': 'E15', 'status': 'complete', 'purpose': 'train_only_descriptive_reference',
                 'run_id': record.run_id, 'created_at': now(), 'policy': POLICY, 'rows': training.num_rows,
                 'scoring_report': str(config['report_path']), 'scoring_report_sha256': sha256_file(config['report_path']),
                 'feature_columns': columns, 'features': fields,
                 'risk_probability': make_spec(scores['risk_probability'].to_pylist(), True),
                 'output_versions': session.bundle['output_versions'], 'original_threshold': session.bundle['threshold'],
                 'source_sha256': {str(session.directory/n): sha256_file(session.directory/n) for n in ['bundle.json','features.parquet','scores.parquet']},
                 'code_sha256': {n: sha256_file(root/n) for n in CODE},
                 'numeric_rule': 'numpy linear training deciles; unique cuts; right-side assignment; separate null; constant <,=,>; all-null nonnull bucket',
                 'notices': NOTICE}
    directory.mkdir(exist_ok=False)
    save_json(directory/'reference.json', reference, exclusive=True)
    record.finish(True, stage='reference_complete', monitoring_reference={'directory':str(directory),
                  'reference_sha256':sha256_file(directory/'reference.json'), 'training_rows':training.num_rows})
    save_json(directory/'completed.json', {'report':str(record.path), 'report_sha256':sha256_file(record.path),
                                         'reference_sha256':sha256_file(directory/'reference.json')}, exclusive=True)
    return reference


def load_reference(root, report_path, scoring_report):
    root, report_path = Path(root).resolve(), Path(report_path).resolve()
    try:
        r = json.loads(report_path.read_text(encoding='utf-8'))
        info = r['monitoring_reference']
        if (not report_path.is_relative_to(root/'reports/runs') or r['status'] != 'passed'
                or r['scope'] != 'official_monitoring_reference' or r['metadata']['module'] != 'E15'):
            raise ValueError('accepted E15 reference required')
        directory = Path(info['directory']).resolve()
        if directory != root/'reports/monitoring/reference'/r['run_id']:
            raise ValueError('reference path')
        path = directory/'reference.json'
        if sha256_file(path) != info['reference_sha256']:
            raise ValueError('reference hash')
        v = json.loads(path.read_text(encoding='utf-8'))
        receipt = json.loads((directory/'completed.json').read_text())
        if (receipt != {'report':str(report_path), 'report_sha256':sha256_file(report_path), 'reference_sha256':info['reference_sha256']}
                or v['version'] != VERSION or v['status'] != 'complete' or v['purpose'] != 'train_only_descriptive_reference'
                or v['run_id'] != r['run_id'] or v['policy'] != POLICY or v['rows'] != info['training_rows']
                or Path(v['scoring_report']).resolve() != Path(scoring_report).resolve()
                or v['scoring_report_sha256'] != sha256_file(scoring_report)
                or set(v['code_sha256']) != set(CODE)
                or any(sha256_file(root/n) != h for n,h in v['code_sha256'].items())
                or any(sha256_file(n) != h for n,h in v['source_sha256'].items())
                or len(v['feature_columns']) != 40 or set(v['features']) != set(v['feature_columns'])):
            raise ValueError('reference receipt, lineage, code or policy mismatch')
        for spec in [*v['features'].values(), v['risk_probability']]:
            counts = spec['reference_counts']
            if len(counts) != spec['buckets'] or any(type(x) is not int or x < 0 for x in counts) or sum(counts) != v['rows']:
                raise ValueError('reference counts')
        return v, info['reference_sha256']
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise DataError('monitor_reference_invalid', 'Complete matching immutable training reference required') from exc


def distribution(spec, counts, n):
    total = sum(spec['reference_counts'])
    result = {'observed_counts':list(counts), 'reference_counts':spec['reference_counts'], 'observations':n,
              'tvd':None, 'missing_rate':None, 'reference_missing_rate':spec['reference_counts'][0]/total,
              'missing_rate_delta':None, 'distribution_review':False, 'missing_review':False}
    if n:
        result['tvd'] = .5*sum(abs(a/n-b/total) for a,b in zip(counts,spec['reference_counts']))
        result['missing_rate'] = counts[0]/n
        result['missing_rate_delta'] = result['missing_rate']-result['reference_missing_rate']
        if n >= POLICY['minimum_observations']:
            result['distribution_review'] = result['tvd'] >= POLICY['tvd_review_threshold']
            result['missing_review'] = abs(result['missing_rate_delta']) >= POLICY['missing_rate_review_threshold']
    return result


class Recorder:
    def __init__(self, reference, reference_sha, path):
        self.reference, self.reference_sha, self.path = reference, reference_sha, Path(path)
        self.file = self.path.open('xb')
        self.lock = threading.RLock()
        self.error = None
        self.events = self.bytes = self.observations = self.successes = 0
        self.statuses, self.errors, self.sample_uses = Counter(), Counter(), Counter()
        self.latency_total_ms = self.latency_max_ms = 0.
        self.latency_count = 0
        self.counts = {n:[0]*s['buckets'] for n,s in reference['features'].items()}
        self.score_counts = [0]*reference['risk_probability']['buckets']

    def record(self, status, payload, session, route, method, elapsed_ms):
        with self.lock:
            if self.error:
                raise DataError(self.error, 'Monitoring unavailable; scoring fails closed')
            if self.events >= POLICY['max_events']:
                self.error = 'monitor_event_limit'
                raise DataError(self.error, 'New monitoring run required; event quota reached')
            rows = payload.get('results', []) if status == 200 else []
            features, score_counts, uses = {}, [], Counter()
            if rows:
                ids = [r['SK_ID_CURR'] for r in rows]
                table = session.features.take(pa.array([session.index[x] for x in ids], type=pa.int64()))
                features = {n:histogram(spec,table[n].to_pylist()) for n,spec in self.reference['features'].items()}
                score_counts = histogram(self.reference['risk_probability'],[r['risk_probability'] for r in rows])
                uses.update(r['sample_use'] for r in rows)
            event = {'version':VERSION, 'sequence':self.events+1, 'at':now(), 'event':'http_response_prepared',
                     'reference_sha256':self.reference_sha, 'scoring_report_sha256':self.reference['scoring_report_sha256'],
                     'route':route if route in {'/v1/score','/v1/score/batch','connection_capacity'} else 'other',
                     'method':method if method in {'GET','POST','HEAD','PUT','DELETE','PATCH','OPTIONS'} else 'OTHER',
                     'status':status, 'error_code':payload.get('error',{}).get('code'),
                     'elapsed_ms':round(max(0.,elapsed_ms),6) if elapsed_ms is not None else None, 'observations':len(rows),
                     'sample_uses':dict(uses), 'feature_counts':features, 'risk_probability_counts':score_counts}
            data = (json.dumps(event, ensure_ascii=False, allow_nan=False, separators=(',',':'))+'\n').encode('utf-8')
            if self.bytes+len(data) > POLICY['max_log_bytes']:
                self.error = 'monitor_byte_limit'
                raise DataError(self.error, 'New monitoring run required; byte quota reached')
            try:
                if self.file.write(data) != len(data):
                    raise OSError('short monitoring write')
                self.file.flush()
                os.fsync(self.file.fileno())
            except (OSError, ValueError) as exc:
                self.error = 'monitor_log_write'
                raise DataError(self.error, 'Monitoring log write failed; scoring fails closed') from exc
            self.bytes += len(data)
            self.events += 1
            self.observations += len(rows)
            self.successes += bool(rows)
            self.statuses[str(status)] += 1
            if event['error_code']:
                self.errors[event['error_code']] += 1
            self.sample_uses.update(uses)
            if event['elapsed_ms'] is not None:
                self.latency_count += 1
                self.latency_total_ms += event['elapsed_ms']
                self.latency_max_ms = max(self.latency_max_ms,event['elapsed_ms'])
            for n,values in features.items():
                self.counts[n] = [a+b for a,b in zip(self.counts[n],values)]
            if score_counts:
                self.score_counts = [a+b for a,b in zip(self.score_counts,score_counts)]
            return event

    def summary(self):
        with self.lock:
            n = self.observations
            return {'version':VERSION, 'reference_sha256':self.reference_sha,
                    'scoring_report_sha256':self.reference['scoring_report_sha256'],
                    'output_versions':self.reference['output_versions'], 'audit_complete':self.error is None,
                    'monitor_error':self.error, 'events':self.events, 'log_bytes':self.bytes,
                    'success_response_events':self.successes, 'observations':n,
                    'statuses':dict(self.statuses), 'errors':dict(self.errors), 'sample_uses':dict(self.sample_uses),
                    'response_preparation_latency_ms':{'count':self.latency_count,'total':self.latency_total_ms,
                        'mean':self.latency_total_ms/self.latency_count if self.latency_count else None,'max':self.latency_max_ms},
                    'distribution_status':'no_observations' if not n else 'insufficient_sample' if n < POLICY['minimum_observations'] else 'descriptive_review_only',
                    'features':{name:distribution(self.reference['features'][name],counts,n) for name,counts in self.counts.items()},
                    'risk_probability':distribution(self.reference['risk_probability'],self.score_counts,n),
                    'policy':POLICY, 'notices':NOTICE}

    def close(self):
        with self.lock:
            self.file.close()


class MonitoredState(ApiState):
    def __init__(self, root, report, recorder):
        super().__init__(root,report)
        self.recorder = recorder

    def health(self):
        result = super().health()
        with self.recorder.lock:
            result.update(monitoring_version=VERSION, monitor_error=self.recorder.error,
                          recorded_events=self.recorder.events, reference_sha256=self.recorder.reference_sha)
        return result


class MonitoredHandler(Handler):
    def setup(self):
        self.monitor_started = time.monotonic()
        super().setup()

    def reply(self, status, value):
        if getattr(self,'path','') not in {'/health/live','/health/ready','/monitor/summary'}:
            try:
                self.server.recorder.record(status,value,self.server.state.session,getattr(self,'path',''),
                                            getattr(self,'command','OTHER'),(time.monotonic()-self.monitor_started)*1000)
            except (DataError, OSError, ValueError, KeyError, TypeError) as exc:
                code = exc.code if isinstance(exc,DataError) else 'monitor_record_failure'
                self.server.recorder.error = code
                self.server.state.phase, self.server.state.error_code = 'failed', code
                status, value = 503,error_payload(code,'Monitoring unavailable; restart after checking this run')
        super().reply(status,value)

    def do_GET(self):
        if self.path == '/monitor/summary':
            self.reply(200,self.server.recorder.summary())
        else:
            super().do_GET()


class MonitoredServer(BoundedServer):
    def __init__(self, state, host, port, recorder):
        if host != '127.0.0.1' or type(port) is not int or not 0 <= port <= 65535 or state.phase != 'ready':
            raise DataError('monitor_bind','Verified loopback-only session required')
        self.recorder = recorder
        super().__init__((host,port),MonitoredHandler,state)

    def process_request(self, request, client_address):
        request.settimeout(API_POLICY['socket_timeout_seconds'])
        if self.slots.acquire(blocking=False):
            try:
                # Reuse inherited worker/finally slot release; avoid acquiring twice.
                ThreadingHTTPServer.process_request(self,request,client_address)
            except BaseException:
                self.slots.release()
                raise
            return
        status, value = 429,error_payload('api_busy','Connection capacity reached; retry later')
        try:
            self.recorder.record(status,value,None,'connection_capacity','OTHER',None)
        except (DataError,OSError,ValueError) as exc:
            code = exc.code if isinstance(exc,DataError) else 'monitor_record_failure'
            self.recorder.error = code
            self.state.phase,self.state.error_code = 'failed',code
            status,value = 503,error_payload(code,'Monitoring unavailable')
        body = json.dumps(value).encode('utf-8')
        try:
            head = f'HTTP/1.0 {status} Response\r\nContent-Type: application/json\r\nCache-Control: no-store\r\nConnection: close\r\nContent-Length: {len(body)}\r\n'
            if status == 429:
                head += 'Retry-After: 1\r\n'
            request.sendall(head.encode()+b'\r\n'+body)
        except (OSError,socket.timeout):
            pass
        finally:
            self.shutdown_request(request)


def load_monitor_run(root, report_path, scoring_report):
    """Read complete stopped success only; failed/torn outputs are not evidence."""
    root, report_path = Path(root).resolve(), Path(report_path).resolve()
    try:
        r = json.loads(report_path.read_text(encoding='utf-8'))
        info = r['monitoring']
        directory = Path(info['directory']).resolve()
        if (not report_path.is_relative_to(root/'reports/runs') or r['status'] != 'passed' or r['stage'] != 'stopped'
                or r['scope'] != 'official_monitored_scoring_api' or r['metadata']['module'] != 'E15'
                or directory != root/'reports/monitoring/runs'/r['run_id']):
            raise ValueError('passed stopped monitoring run required')
        ref,digest = load_reference(root,info['reference_report'],scoring_report)
        if digest != info['reference_sha256'] or set(r['output_sha256']) != {'events.jsonl','summary.json'}:
            raise ValueError('reference/output mismatch')
        if any(sha256_file(directory/n) != h for n,h in r['output_sha256'].items()):
            raise ValueError('changed output')
        receipt = json.loads((directory/'completed.json').read_text())
        expected = {'report':str(report_path),'report_sha256':sha256_file(report_path),'status':'passed','outputs':r['output_sha256']}
        if receipt != expected:
            raise ValueError('completion receipt mismatch')
        summary = json.loads((directory/'summary.json').read_text())
        raw = (directory/'events.jsonl').read_bytes()
        if ((raw and not raw.endswith(b'\n')) or not summary['audit_complete'] or summary['monitor_error'] is not None
                or summary['reference_sha256'] != digest or len(raw) != summary['log_bytes']):
            raise ValueError('incomplete log or summary')
        events = [json.loads(line) for line in raw.splitlines()]
        if len(events) != summary['events'] or any(e['sequence'] != i+1 for i,e in enumerate(events)):
            raise ValueError('event sequence mismatch')
        return {'report':r,'directory':directory,'reference':ref,'events':events,'summary':summary}
    except (OSError,ValueError,KeyError,TypeError) as exc:
        raise DataError('monitor_run_invalid','Complete unchanged stopped monitoring output required') from exc
