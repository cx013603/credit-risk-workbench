"""Sealed, read-only SQLite history evidence; no model or data-engine imports."""
import hashlib
import json
from contextlib import closing
from pathlib import Path
import sqlite3

from .contracts import integer, population, require, revision

VERSION = 'workbench_history_index_v1'
POLICY = 'history_before_day_zero_currency_separate_v1'
FIELDS = ['BUREAU_ELIGIBLE_COUNT', 'BUREAU_HAS_ELIGIBLE_HISTORY', 'BUREAU_DAYS_SINCE_MOST_RECENT',
          'BUREAU_C1_COUNT', 'BUREAU_OTHER_CURRENCY_COUNT', 'BUREAU_CURRENCY_MISSING_COUNT',
          'BUREAU_C1_AMOUNT_AVAILABLE_COUNT', 'BUREAU_C1_AMOUNT_UNAVAILABLE_COUNT',
          'BUREAU_C1_CREDIT_SUM', 'BUREAU_C1_CREDIT_MEAN', 'BUREAU_C1_CREDIT_MAX',
          'BUREAU_ACTIVE_COUNT', 'BUREAU_CLOSED_COUNT', 'BUREAU_SOLD_COUNT', 'BUREAU_BAD_DEBT_COUNT',
          'BUREAU_OTHER_STATUS_COUNT', 'BUREAU_ACTIVE_FRACTION', 'PREV_ELIGIBLE_COUNT',
          'PREV_HAS_ELIGIBLE_HISTORY', 'PREV_DAYS_SINCE_MOST_RECENT', 'PREV_CASH_COUNT',
          'PREV_CONSUMER_COUNT', 'PREV_REVOLVING_COUNT', 'PREV_OTHER_CONTRACT_COUNT']
COLUMNS = {
    'bureau': ['SK_ID_CURR', 'SK_ID_BUREAU', 'DAYS_CREDIT', 'DAYS_CREDIT_UPDATE', 'CREDIT_CURRENCY', 'CREDIT_ACTIVE', 'AMT_CREDIT_SUM'],
    'previous': ['SK_ID_CURR', 'SK_ID_PREV', 'DAYS_DECISION', 'NAME_CONTRACT_TYPE'],
}
KEYS = {'bureau': 'SK_ID_BUREAU', 'previous': 'SK_ID_PREV'}
REASONS = {'eligible': '符合时间准入规则', 'missing_credit_time': '借款日缺失',
           'credit_not_before': '借款日为同日或未来', 'missing_update_time': '更新日缺失',
           'update_not_before': '更新日为同日或未来', 'missing_decision_time': '决策日缺失',
           'decision_not_before': '决策日为同日或未来'}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def fingerprint(values):
    return hashlib.sha256(json.dumps(values, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def classify(table, row):
    if table == 'previous':
        value = row['DAYS_DECISION']
        return 'missing_decision_time' if value is None else 'decision_not_before' if value >= 0 else 'eligible'
    for name, missing, future in [('DAYS_CREDIT', 'missing_credit_time', 'credit_not_before'),
                                  ('DAYS_CREDIT_UPDATE', 'missing_update_time', 'update_not_before')]:
        if row[name] is None:
            return missing
        if row[name] >= 0:
            return future
    return 'eligible'


def amount_state(row):
    if row['CREDIT_CURRENCY'] is None:
        return 'currency_missing'
    if row['CREDIT_CURRENCY'] != 'currency 1':
        return 'other_currency_not_summed'
    if row['AMT_CREDIT_SUM'] is None:
        return 'amount_missing'
    return 'amount_negative' if row['AMT_CREDIT_SUM'] < 0 else 'currency1_nonnegative'


def input_predicate(feature):
    """Inputs to a feature formula, not causal/model contribution attribution."""
    require(feature in FIELDS)
    if feature in ['BUREAU_C1_COUNT', 'BUREAU_C1_AMOUNT_AVAILABLE_COUNT', 'BUREAU_C1_AMOUNT_UNAVAILABLE_COUNT',
                   'BUREAU_C1_CREDIT_SUM', 'BUREAU_C1_CREDIT_MEAN', 'BUREAU_C1_CREDIT_MAX']:
        extra = "CREDIT_CURRENCY='currency 1'"
        if feature == 'BUREAU_C1_AMOUNT_UNAVAILABLE_COUNT':
            extra += ' AND (AMT_CREDIT_SUM IS NULL OR AMT_CREDIT_SUM<0)'
        elif feature != 'BUREAU_C1_COUNT':
            extra += ' AND AMT_CREDIT_SUM>=0'
    elif feature == 'BUREAU_OTHER_CURRENCY_COUNT':
        extra = "CREDIT_CURRENCY<>'currency 1'"
    elif feature == 'BUREAU_CURRENCY_MISSING_COUNT':
        extra = 'CREDIT_CURRENCY IS NULL'
    elif feature in ['BUREAU_ACTIVE_COUNT', 'BUREAU_CLOSED_COUNT', 'BUREAU_SOLD_COUNT', 'BUREAU_BAD_DEBT_COUNT']:
        extra = 'CREDIT_ACTIVE=' + {'BUREAU_ACTIVE_COUNT': "'Active'", 'BUREAU_CLOSED_COUNT': "'Closed'",
                                    'BUREAU_SOLD_COUNT': "'Sold'", 'BUREAU_BAD_DEBT_COUNT': "'Bad debt'"}[feature]
    elif feature == 'BUREAU_OTHER_STATUS_COUNT':
        extra = "(CREDIT_ACTIVE IS NULL OR CREDIT_ACTIVE NOT IN ('Active','Closed','Sold','Bad debt'))"
    elif feature in ['PREV_CASH_COUNT', 'PREV_CONSUMER_COUNT', 'PREV_REVOLVING_COUNT']:
        extra = 'NAME_CONTRACT_TYPE=' + {'PREV_CASH_COUNT': "'Cash loans'", 'PREV_CONSUMER_COUNT': "'Consumer loans'",
                                        'PREV_REVOLVING_COUNT': "'Revolving loans'"}[feature]
    elif feature == 'PREV_OTHER_CONTRACT_COUNT':
        extra = "(NAME_CONTRACT_TYPE IS NULL OR NAME_CONTRACT_TYPE NOT IN ('Cash loans','Consumer loans','Revolving loans'))"
    else:
        extra = '1'
    return "exclusion_reason='eligible' AND (" + extra + ')'


class HistoryError(ValueError):
    pass


def ro_connection(path):
    connection = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro&immutable=1', uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute('PRAGMA query_only=ON')
    connection.execute('PRAGMA cache_size=-8192')
    return connection


class HistoryIndex:
    def __init__(self, manifest_path, seal, release_seal, dictionary_hash):
        self.path = Path(manifest_path).resolve()
        try:
            if digest(self.path) != seal:
                raise ValueError('manifest fingerprint')
            self.manifest = json.loads(self.path.read_text(encoding='utf-8'))
            m = self.manifest
            if (m['version'] != VERSION or m['status'] != 'passed' or m['policy'] != POLICY or m['fields'] != FIELDS
                    or m['release_manifest_sha256'] != release_seal or m['dictionary_sha256'] != dictionary_hash
                    or m['validation']['saved_feature_mismatches'] != 0 or m['validation']['sqlite_feature_mismatches'] != 0
                    or m['database']['name'] != 'history.sqlite'):
                raise ValueError('version/validation binding')
            self.database = self.path.parent / 'history.sqlite'
            if self.database.stat().st_size != m['database']['bytes'] or digest(self.database) != m['database']['sha256']:
                raise ValueError('database fingerprint')
            with closing(ro_connection(self.database)) as connection:
                if connection.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
                    raise ValueError('database integrity')
                if dict(connection.execute('SELECT key,value FROM metadata')) != {'version': VERSION, 'policy': POLICY}:
                    raise ValueError('metadata')
            self.seal, self.release_seal, self.dictionary_hash = seal, release_seal, dictionary_hash
            self.database_stat = self.database.stat()
        except (OSError, KeyError, TypeError, ValueError, sqlite3.Error) as exc:
            raise HistoryError('HISTORY_INDEX_INVALID') from exc

    def info(self):
        return {'status': 'ready', 'version': VERSION, 'index_manifest_sha256': self.seal, 'policy': POLICY,
                'release_manifest_sha256': self.release_seal, 'dictionary_sha256': self.dictionary_hash,
                'max_page': 50, 'allowed_uses': ['train', 'validation', 'unlabeled_scoring'],
                'source_fingerprints': self.manifest['sources'], 'validation': self.manifest['validation']}

    def query(self, pop, ident, table, feature=None, selection='eligible', offset=0, limit=10, expected=None):
        population(pop); integer(ident); integer(offset, 0, 10**7); integer(limit, 1, 50)
        require(table in COLUMNS and selection in ['eligible', 'excluded', 'all', 'inputs'])
        require(feature is None or feature in FIELDS and feature.startswith('BUREAU_' if table == 'bureau' else 'PREV_'))
        require(selection != 'inputs' or feature is not None)
        current_revision = revision(self.release_seal, self.dictionary_hash, pop, ident)
        try:
            stat = self.database.stat()
            if (stat.st_size, stat.st_mtime_ns, stat.st_ino) != (self.database_stat.st_size, self.database_stat.st_mtime_ns, self.database_stat.st_ino):
                raise HistoryError('HISTORY_INDEX_UNAVAILABLE')
            with closing(ro_connection(self.database)) as con:
                case = con.execute('SELECT * FROM cases WHERE population=? AND application_id=?', (pop, ident)).fetchone()
                if case is None:
                    denied = pop == 'labeled' and con.execute('SELECT 1 FROM denied WHERE application_id=?', (ident,)).fetchone()
                    raise HistoryError('SAMPLE_USE_NOT_ALLOWED' if denied else 'CASE_NOT_FOUND')
                require(expected is None or expected == current_revision, 'VERSION_MISMATCH')
                totals = dict(con.execute(f'SELECT exclusion_reason,COUNT(*) FROM {table} WHERE SK_ID_CURR=? GROUP BY exclusion_reason', (ident,)))
                raw = sum(totals.values()); eligible = totals.get('eligible', 0)
                predicate = input_predicate(feature) if feature else "exclusion_reason='eligible'"
                conditions = {'eligible': "exclusion_reason='eligible'", 'excluded': "exclusion_reason<>'eligible'", 'all': '1', 'inputs': predicate}
                where = 'SK_ID_CURR=? AND (' + conditions[selection] + ')'
                total = con.execute(f'SELECT COUNT(*) FROM {table} WHERE ' + where, (ident,)).fetchone()[0]
                rows = con.execute(f'SELECT *,COALESCE(({predicate}),0) AS feature_input_used FROM {table} WHERE {where} ORDER BY {KEYS[table]} LIMIT ? OFFSET ?', (ident, limit, offset)).fetchall()
                inputs_count = con.execute(f'SELECT COUNT(*) FROM {table} WHERE SK_ID_CURR=? AND ({predicate})', (ident,)).fetchone()[0]
                items = []
                for row in rows:
                    values = {name: row[name] for name in COLUMNS[table]}
                    items.append({'source_key': row[KEYS[table]], 'normalized_projection_sha256': row['fingerprint'],
                                  'source_values': values, 'eligible': row['exclusion_reason'] == 'eligible',
                                  'exclusion_reason': row['exclusion_reason'], 'reason_text': REASONS[row['exclusion_reason']],
                                  'feature_input_used': bool(row['feature_input_used']),
                                  'amount_state': amount_state(values) if table == 'bureau' else 'not_used_by_original_features'})
                values = dict(zip(FIELDS, json.loads(case['values_json'])))
                return {'case_key': {'population': pop, 'application_id': ident}, 'case_revision': current_revision,
                        'sample_use': case['sample_use'], 'table': table, 'feature': feature, 'selection': selection,
                        'index_version': VERSION, 'index_manifest_sha256': self.seal, 'policy': POLICY,
                        'time_reference': 'day_zero_is_current_application; relative_days_only; within_day_order_unknown',
                        'source': self.manifest['sources'][table], 'summary_values': {k: v for k, v in values.items() if k.startswith('BUREAU_' if table == 'bureau' else 'PREV_')},
                        'audit': {'raw_count': raw, 'eligible_count': eligible, 'excluded_count': raw-eligible,
                                  'reason_counts': totals, 'feature_inputs_count': inputs_count,
                                  'history_state': 'eligible_present' if eligible else 'only_time_excluded' if raw else 'no_source_history',
                                  'role': 'audit_only_never_model_input'},
                        'items': items, 'total': total, 'offset': offset, 'limit': limit,
                        'next_offset': offset+limit if offset+limit < total else None,
                        'order': 'source_key_ascending', 'amount_note': 'currency1 is anonymous; no cross-currency sum; previous amounts/status not used',
                        'limits': ['Normalized input projection, not a raw CSV byte-row hash', 'No absolute calendar dates, natural-person reconstruction or payment trend',
                                   'Inputs to aggregation formula are separate from tree path contributions']}
        except (OSError, sqlite3.Error) as exc:
            raise HistoryError('HISTORY_INDEX_UNAVAILABLE') from exc
