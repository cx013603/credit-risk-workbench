"""Synthetic in-memory walkthrough using the original aggregation SQL; no fit."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import duckdb
from credit_risk.history import FEATURES, COUNT_NAMES


def main():
    with duckdb.connect(':memory:') as db:
        db.execute('CREATE TABLE application_input(SK_ID_CURR BIGINT, file_row_number BIGINT)')
        db.executemany('INSERT INTO application_input VALUES (?, ?)', [(i, i) for i in range(1, 5)])
        db.execute('CREATE TABLE bureau_input(SK_ID_CURR BIGINT, SK_ID_BUREAU BIGINT, DAYS_CREDIT DOUBLE, DAYS_CREDIT_UPDATE DOUBLE, CREDIT_CURRENCY VARCHAR, CREDIT_ACTIVE VARCHAR, AMT_CREDIT_SUM DOUBLE)')
        db.executemany('INSERT INTO bureau_input VALUES (?, ?, ?, ?, ?, ?, ?)', [
            (1, 11, -20, -1, 'currency 1', 'Active', 100),
            (1, 12, -10, -1, 'currency 1', 'Closed', 200),
            (1, 13, 1, -1, 'currency 1', 'Active', 99999),
            (2, 21, -5, -1, 'currency 1', 'Active', None),
            (4, 41, -5, -1, 'currency 1', 'Active', 0),
        ])
        db.execute('CREATE TABLE previous_input(SK_ID_CURR BIGINT, SK_ID_PREV BIGINT, DAYS_DECISION DOUBLE, NAME_CONTRACT_TYPE VARCHAR)')
        db.executemany('INSERT INTO previous_input VALUES (?, ?, ?, ?)', [
            (1, 101, -4, 'Cash loans'), (1, 102, -2, 'Consumer loans'),
            (1, 103, 0, 'Cash loans'),
        ])
        for name in ('02_bureau_features.sql', '03_previous_features.sql'):
            db.execute((ROOT / 'sql' / name).read_text(encoding='utf-8'))
        bad_rows, bad_sum = db.execute('SELECT COUNT(*), SUM(b.AMT_CREDIT_SUM) FROM bureau_eligible b JOIN previous_eligible p USING(SK_ID_CURR) WHERE b.SK_ID_CURR=1').fetchone()
        assert (bad_rows, bad_sum) == (4, 600)
        columns = []
        for name, *_ in FEATURES:
            alias = 'b' if name.startswith('BUREAU_') else 'p'
            expr = f'{alias}."{name}"'
            if name in COUNT_NAMES:
                expr = f'COALESCE({expr},0)'
            columns.append(f'{expr} AS "{name}"')
        query = (ROOT / 'sql/04_join_history.sql').read_text(encoding='utf-8').replace('{{history_columns}}', ', '.join(columns))
        cursor = db.execute(query)
        names = [col[0] for col in cursor.description]
        rows = [dict(zip(names, row)) for row in cursor.fetchall()]
        assert len(rows) == 4 and len(names) == 25
        assert [row['SK_ID_CURR'] for row in rows] == [1, 2, 3, 4]
        assert rows[0]['BUREAU_ELIGIBLE_COUNT'] == rows[0]['PREV_ELIGIBLE_COUNT'] == 2
        assert rows[0]['BUREAU_C1_CREDIT_SUM'] == 300
        assert rows[1]['BUREAU_ELIGIBLE_COUNT'] == 1 and rows[1]['BUREAU_C1_CREDIT_SUM'] is None
        assert rows[2]['BUREAU_ELIGIBLE_COUNT'] == 0 and rows[2]['BUREAU_C1_CREDIT_SUM'] is None
        assert rows[3]['BUREAU_ELIGIBLE_COUNT'] == 1 and rows[3]['BUREAU_C1_CREDIT_SUM'] == 0
        print(f'Synthetic application 1: direct join {bad_rows} rows, wrong amount {bad_sum:.0f}; aggregate first: 1 row, correct amount 300.')
        show = ['SK_ID_CURR', 'BUREAU_ELIGIBLE_COUNT', 'PREV_ELIGIBLE_COUNT', 'BUREAU_C1_CREDIT_SUM']
        print(json.dumps([{k: row[k] for k in show} for row in rows], indent=2))
        print('PASS: 4 synthetic applications, 24 history features; future/day-zero excluded; null and observed zero remain distinct.')


if __name__ == '__main__':
    main()
