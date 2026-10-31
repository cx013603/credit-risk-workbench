"""Offline index builder: accepted E03/E04 inputs, original SQL, no models."""
import json
import math
from pathlib import Path
import sqlite3
import time
import uuid

import duckdb

from credit_risk.normalize import load_normalized_bundle
from credit_risk.history import load_history_bundle, FEATURE_NAMES, COUNT_NAMES
from .history_index import VERSION, POLICY, FIELDS, COLUMNS, KEYS, digest, fingerprint

NORMAL_RUN = 'reports/runs/20261025-140410-7b969f0a/run.json'
HISTORY_RUN = 'reports/runs/20261025-145716-9f2f2a4d/run.json'
FEATURE_FILES = {'labeled': 'project/reports/scoring/full/20261025-211906-8423ca02/features.parquet',
                 'unlabeled': 'project/reports/unlabeled_scoring/20261027-140524-1fc773ac/features.parquet'}


def literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def compare_values(left, right):
    return all(a is None and b is None or a is not None and b is not None and
               (a == b if name in COUNT_NAMES else math.isclose(a, b, abs_tol=1e-9, rel_tol=1e-12))
               for name, a, b in zip(FIELDS, left, right))


# Independent SQLite aggregate expressions, separate from the accepted DuckDB SQL.
def verify_sqlite(database):
    from contextlib import closing
    from .history_index import ro_connection
    with closing(ro_connection(database)) as con:
        b = """SELECT SK_ID_CURR,COUNT(*) AS BUREAU_ELIGIBLE_COUNT,1 AS BUREAU_HAS_ELIGIBLE_HISTORY,
        -MAX(DAYS_CREDIT) AS BUREAU_DAYS_SINCE_MOST_RECENT,
        SUM(CASE WHEN CREDIT_CURRENCY='currency 1' THEN 1 ELSE 0 END) AS BUREAU_C1_COUNT,
        SUM(CASE WHEN CREDIT_CURRENCY<>'currency 1' THEN 1 ELSE 0 END) AS BUREAU_OTHER_CURRENCY_COUNT,
        SUM(CASE WHEN CREDIT_CURRENCY IS NULL THEN 1 ELSE 0 END) AS BUREAU_CURRENCY_MISSING_COUNT,
        SUM(CASE WHEN CREDIT_CURRENCY='currency 1' AND AMT_CREDIT_SUM>=0 THEN 1 ELSE 0 END) AS BUREAU_C1_AMOUNT_AVAILABLE_COUNT,
        SUM(CASE WHEN CREDIT_CURRENCY='currency 1' AND (AMT_CREDIT_SUM IS NULL OR AMT_CREDIT_SUM<0) THEN 1 ELSE 0 END) AS BUREAU_C1_AMOUNT_UNAVAILABLE_COUNT,
        SUM(CASE WHEN CREDIT_CURRENCY='currency 1' AND AMT_CREDIT_SUM>=0 THEN AMT_CREDIT_SUM END) AS BUREAU_C1_CREDIT_SUM,
        AVG(CASE WHEN CREDIT_CURRENCY='currency 1' AND AMT_CREDIT_SUM>=0 THEN AMT_CREDIT_SUM END) AS BUREAU_C1_CREDIT_MEAN,
        MAX(CASE WHEN CREDIT_CURRENCY='currency 1' AND AMT_CREDIT_SUM>=0 THEN AMT_CREDIT_SUM END) AS BUREAU_C1_CREDIT_MAX,
        SUM(CASE WHEN CREDIT_ACTIVE='Active' THEN 1 ELSE 0 END) AS BUREAU_ACTIVE_COUNT,
        SUM(CASE WHEN CREDIT_ACTIVE='Closed' THEN 1 ELSE 0 END) AS BUREAU_CLOSED_COUNT,
        SUM(CASE WHEN CREDIT_ACTIVE='Sold' THEN 1 ELSE 0 END) AS BUREAU_SOLD_COUNT,
        SUM(CASE WHEN CREDIT_ACTIVE='Bad debt' THEN 1 ELSE 0 END) AS BUREAU_BAD_DEBT_COUNT,
        SUM(CASE WHEN CREDIT_ACTIVE IS NULL OR CREDIT_ACTIVE NOT IN ('Active','Closed','Sold','Bad debt') THEN 1 ELSE 0 END) AS BUREAU_OTHER_STATUS_COUNT,
        1.0*SUM(CASE WHEN CREDIT_ACTIVE='Active' THEN 1 ELSE 0 END)/COUNT(*) AS BUREAU_ACTIVE_FRACTION
        FROM bureau WHERE exclusion_reason='eligible' GROUP BY SK_ID_CURR"""
        p = """SELECT SK_ID_CURR,COUNT(*) AS PREV_ELIGIBLE_COUNT,1 AS PREV_HAS_ELIGIBLE_HISTORY,
        -MAX(DAYS_DECISION) AS PREV_DAYS_SINCE_MOST_RECENT,
        SUM(CASE WHEN NAME_CONTRACT_TYPE='Cash loans' THEN 1 ELSE 0 END) AS PREV_CASH_COUNT,
        SUM(CASE WHEN NAME_CONTRACT_TYPE='Consumer loans' THEN 1 ELSE 0 END) AS PREV_CONSUMER_COUNT,
        SUM(CASE WHEN NAME_CONTRACT_TYPE='Revolving loans' THEN 1 ELSE 0 END) AS PREV_REVOLVING_COUNT,
        SUM(CASE WHEN NAME_CONTRACT_TYPE IS NULL OR NAME_CONTRACT_TYPE NOT IN ('Cash loans','Consumer loans','Revolving loans') THEN 1 ELSE 0 END) AS PREV_OTHER_CONTRACT_COUNT
        FROM previous WHERE exclusion_reason='eligible' GROUP BY SK_ID_CURR"""
        columns = ','.join(f"COALESCE({'b' if name.startswith('BUREAU_') else 'p'}.{name},0)" if name in COUNT_NAMES
                           else f"{'b' if name.startswith('BUREAU_') else 'p'}.{name}" for name in FIELDS)
        cursor = con.execute(f'WITH b AS ({b}),p AS ({p}) SELECT c.values_json,{columns} FROM cases c LEFT JOIN b ON c.application_id=b.SK_ID_CURR LEFT JOIN p ON c.application_id=p.SK_ID_CURR ORDER BY c.application_id')
        count = mismatches = 0
        for row in cursor:
            count += 1
            if not compare_values(json.loads(row[0]), row[1:]):
                mismatches += 1
        return {'cases_checked': count, 'sqlite_feature_mismatches': mismatches,
                'numeric_tolerance': {'absolute': 1e-9, 'relative': 1e-12}, 'count_null_semantics': 'exact'}


def build(root, release, output):
    root, release, output = Path(root).resolve(), Path(release).resolve(), Path(output).resolve()
    if output.exists() or not output.is_relative_to(root / 'data/interim/workbench_history'):
        raise ValueError('New dedicated output under data/interim/workbench_history required')
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.parent / ('.partial-' + uuid.uuid4().hex)
    stage.mkdir()
    started = time.monotonic()
    con = target = None
    try:
        normalized = load_normalized_bundle(root, root / NORMAL_RUN)
        accepted = load_history_bundle(root, root / HISTORY_RUN)
        config = json.loads((root / 'config/workbench/api.json').read_text(encoding='utf-8'))
        seal = digest(release / 'portable_manifest.json')
        if seal != config['release_manifest_sha256'] or digest(root / config['dictionary']) != config['dictionary_sha256']:
            raise ValueError('Frozen release/dictionary binding differs')
        manifest = json.loads((release / 'portable_manifest.json').read_text(encoding='utf-8'))
        sources = {}
        for pop, path in FEATURE_FILES.items():
            if digest(release / path) != manifest['files'][path]['sha256']:
                raise ValueError('Saved feature fingerprint differs')
        if FIELDS != FEATURE_NAMES:
            raise ValueError('History field contract differs')
        for name in ['sql/02_bureau_features.sql', 'sql/03_previous_features.sql']:
            if digest(root / name) != accepted['bundle']['sql_sha256'][name]:
                raise ValueError('Original SQL fingerprint differs')
        con = duckdb.connect()
        con.execute("SET threads=1; SET memory_limit='512MB'")
        con.execute('SET temp_directory=' + literal(stage / 'duck-temp'))
        projection = ','.join(FIELDS)
        for pop, path in FEATURE_FILES.items():
            con.execute(f"CREATE TEMP VIEW {pop}_all AS SELECT SK_ID_CURR,sample_use FROM read_parquet({literal(release / path)})")
            use_filter = "IN ('train','validation')" if pop == 'labeled' else "='unlabeled_scoring'"
            con.execute(f"CREATE TEMP VIEW {pop}_allowed AS SELECT SK_ID_CURR,sample_use,{projection} FROM read_parquet({literal(release / path)}) WHERE sample_use {use_filter}")
        con.execute("CREATE TABLE allowed AS SELECT 'labeled' AS population,* FROM labeled_allowed UNION ALL SELECT 'unlabeled',* FROM unlabeled_allowed")
        if con.execute('SELECT COUNT(*)-COUNT(DISTINCT SK_ID_CURR) FROM allowed').fetchone()[0]:
            raise ValueError('Population identities collide')
        for table, filename in [('bureau', 'bureau.csv'), ('previous', 'previous_application.csv')]:
            path = normalized['tables'][filename]
            columns = ','.join('h.' + name for name in COLUMNS[table])
            con.execute(f'CREATE TEMP VIEW {table}_input AS SELECT {columns} FROM read_parquet({literal(path)}) h INNER JOIN allowed a USING(SK_ID_CURR)')
            sql = (root / ('sql/02_bureau_features.sql' if table == 'bureau' else 'sql/03_previous_features.sql')).read_text(encoding='utf-8')
            con.execute(sql)
            sources[table] = {'table': filename, 'normalized_parquet_sha256': digest(path),
                              'normalization_run': Path(NORMAL_RUN).parent.name,
                              'normalization_bundle_sha256': digest(normalized['directory'] / 'bundle.json'),
                              'original_sql_sha256': digest(root / ('sql/02_bureau_features.sql' if table == 'bureau' else 'sql/03_previous_features.sql')),
                              'columns': COLUMNS[table], 'record_key': KEYS[table],
                              'projection_fingerprint': 'SHA256 canonical JSON array in columns order; normalized values, not raw CSV bytes'}
        actual = ','.join(f"COALESCE({'b' if n.startswith('BUREAU_') else 'p'}.{n},0)" if n in COUNT_NAMES else f"{'b' if n.startswith('BUREAU_') else 'p'}.{n}" for n in FIELDS)
        wanted = ','.join('a.' + n for n in FIELDS)
        cursor = con.execute(f'SELECT a.population,a.SK_ID_CURR,a.sample_use,{wanted},{actual} FROM allowed a LEFT JOIN bureau_features b USING(SK_ID_CURR) LEFT JOIN previous_features p USING(SK_ID_CURR) ORDER BY a.SK_ID_CURR')
        target = sqlite3.connect(stage / 'history.sqlite')
        target.execute('PRAGMA journal_mode=DELETE'); target.execute('PRAGMA cache_size=-32768')
        target.executescript('CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL); CREATE TABLE cases(population TEXT NOT NULL,application_id INTEGER NOT NULL,sample_use TEXT NOT NULL,values_json TEXT NOT NULL,PRIMARY KEY(population,application_id)) WITHOUT ROWID; CREATE TABLE denied(application_id INTEGER PRIMARY KEY);')
        target.executemany('INSERT INTO metadata VALUES (?,?)', [('version', VERSION), ('policy', POLICY)])
        mismatch = cases = 0
        while rows := cursor.fetchmany(4096):
            records = []
            for row in rows:
                cases += 1
                left, right = row[3:27], row[27:51]
                if not compare_values(left, right):
                    mismatch += 1
                records.append((*row[:3], json.dumps(left, allow_nan=False, separators=(',', ':'))))
            target.executemany('INSERT INTO cases VALUES (?,?,?,?)', records)
        if mismatch:
            raise ValueError(f'Original SQL / saved24 mismatch: {mismatch}')
        target.executemany('INSERT INTO denied VALUES (?)', con.execute("SELECT SK_ID_CURR FROM labeled_all WHERE sample_use='test'").fetchall())
        row_counts = {}
        for table in COLUMNS:
            definitions = ','.join(name + (' INTEGER NOT NULL' if name.startswith('SK_ID_') else ' TEXT' if name in ['CREDIT_CURRENCY','CREDIT_ACTIVE','NAME_CONTRACT_TYPE'] else ' REAL') for name in COLUMNS[table])
            target.execute(f'CREATE TABLE {table} ({definitions},exclusion_reason TEXT NOT NULL,fingerprint TEXT NOT NULL,PRIMARY KEY(SK_ID_CURR,{KEYS[table]})) WITHOUT ROWID')
            cols = ','.join(COLUMNS[table])
            cursor = con.execute(f'SELECT {cols},exclusion_reason FROM {table}_classified ORDER BY SK_ID_CURR,{KEYS[table]}')
            count = 0
            while rows := cursor.fetchmany(8192):
                target.executemany(f'INSERT INTO {table} VALUES ({",".join("?" for _ in range(len(COLUMNS[table])+2))})',
                                   [(*row, fingerprint(list(row[:-1]))) for row in rows])
                count += len(rows)
            row_counts[table] = count
            target.execute(f'CREATE INDEX {table}_reason ON {table}(SK_ID_CURR,exclusion_reason,{KEYS[table]})')
            target.commit()
            print(f'{table}: indexed {count} allowed source records', flush=True)
        target.execute('ANALYZE'); target.commit(); target.close(); target = None
        verify = verify_sqlite(stage / 'history.sqlite')
        if verify['sqlite_feature_mismatches'] or verify['cases_checked'] != cases:
            raise ValueError('Persisted SQLite aggregation differs')
        populations = dict(con.execute('SELECT sample_use,COUNT(*) FROM allowed GROUP BY sample_use').fetchall())
        for pop, path in FEATURE_FILES.items():
            if digest(release / path) != manifest['files'][path]['sha256']:
                raise ValueError('Saved features changed during build')
        for table, filename in [('bureau', 'bureau.csv'), ('previous', 'previous_application.csv')]:
            if digest(normalized['tables'][filename]) != sources[table]['normalized_parquet_sha256']:
                raise ValueError('Normalized source changed during build')
        result = {'version': VERSION, 'status': 'passed', 'policy': POLICY, 'fields': FIELDS,
                  'release_manifest_sha256': seal, 'dictionary_sha256': config['dictionary_sha256'],
                  'sources': sources, 'source_feature_files': {pop: {'path': path, 'sha256': manifest['files'][path]['sha256']} for pop, path in FEATURE_FILES.items()},
                  'original_E04_bundle_sha256': digest(accepted['directory'] / 'bundle.json'),
                  'database': {'name': 'history.sqlite', 'bytes': (stage / 'history.sqlite').stat().st_size, 'sha256': digest(stage / 'history.sqlite')},
                  'counts': {'cases': cases, 'sample_uses': populations, 'source_records': row_counts,
                             'denied_test_ids': con.execute("SELECT COUNT(*) FROM labeled_all WHERE sample_use='test'").fetchone()[0]},
                  'validation': {'saved_feature_mismatches': mismatch, **verify},
                  'build_seconds': round(time.monotonic()-started, 3), 'resource_budget': 'DuckDB1thread/512MB; SQLite32MiB cache; row batches8192',
                  'scope': 'training/validation/unlabeled existing IDs only; no labels/model loads/fitting/new effect evaluation',
                  'code_sha256': {str(p.relative_to(root)): digest(p) for p in [root/'src/credit_risk_workbench/build_history_index.py',root/'src/credit_risk_workbench/history_index.py']}}
        (stage / 'manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        con.close(); con = None
        # Remove only an empty, resolved staging-owned temporary directory.
        temporary = stage / 'duck-temp'
        if temporary.exists() and temporary.resolve().is_relative_to(stage.resolve()) and not any(temporary.iterdir()):
            temporary.rmdir()
        stage.rename(output)
        return result
    except Exception:
        if target:
            target.close()
        if con:
            con.close()
        # Retain failed staging evidence; never publish it or delete an existing index.
        (stage / 'failed.json').write_text(json.dumps({'status': 'failed', 'scope': 'not_published'}), encoding='utf-8')
        raise
