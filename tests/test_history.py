"""E04 手算、时间反例、空历史、粒度守恒、事务恢复与 E03→E04→E05 接线；仅人工夹具。"""

import copy
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pyarrow.parquet as pq
from credit_risk.history import build_history_features, feature_contract, load_history_bundle, load_history_config, FEATURE_NAMES
from credit_risk.normalize import load_normalization_config, normalize_dataset
from credit_risk.raw import DataError, RawDataSource, sha256_file
from credit_risk.runtime import load_config, RunRecord

TRAIN, BUREAU, PREV = "application_train.csv", "bureau.csv", "previous_application.csv"


class HistoryTests(unittest.TestCase):
    def setUp(self):
        parent = (ROOT / 'data/interim').resolve()
        parent.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=parent)
        self.root = Path(self.temp.name).resolve()
        assert self.root.is_relative_to(parent)
        self.addCleanup(self.temp.cleanup)
        report_parent = (ROOT / 'reports').resolve()
        self.report_temp = tempfile.TemporaryDirectory(dir=report_parent)
        self.reports = Path(self.report_temp.name).resolve()
        assert self.reports.is_relative_to(report_parent)
        self.addCleanup(self.report_temp.cleanup)
        self.raw = self.root / 'raw'
        self.raw.mkdir()
        self.contents = {
            TRAIN: 'SK_ID_CURR,TARGET,AMT_INCOME_TOTAL,CODE_GENDER\n1,0,100,XNA\n2,1,,NULL\n3,0,-1,F\n4,1,200,M\n',
            'application_test.csv': 'SK_ID_CURR\n5\n6\n',
            BUREAU: 'SK_ID_CURR,SK_ID_BUREAU,DAYS_CREDIT,DAYS_CREDIT_UPDATE,CREDIT_CURRENCY,CREDIT_ACTIVE,AMT_CREDIT_SUM\n'
                    '1,10,-20,-2,currency 1,Active,100\n1,11,-5,-1,currency 1,Closed,\n1,12,-999,-1,currency 2,Sold,900\n'
                    '1,13,-3,0,currency 1,Active,10000\n1,14,1,-1,currency 1,Active,20000\n'
                    '2,15,-2,-1,currency 1,XNA,-999\n4,16,-5,5,currency 1,Active,400\n5,17,-1,-1,currency 1,Active,30\n',
            PREV: 'SK_ID_PREV,SK_ID_CURR,DAYS_DECISION,NAME_CONTRACT_TYPE,NAME_CONTRACT_STATUS,AMT_APPLICATION,DAYS_LAST_DUE,DAYS_TERMINATION\n'
                  '100,1,-2,Cash loans,Approved,100,365243,365243\n101,1,-10,Consumer loans,Refused,999,100,200\n'
                  '102,1,0,Revolving loans,Approved,999999,1000,2000\n103,2,-1,XNA,Unknown,-999,,\n'
                  '104,4,1,Consumer loans,Approved,200,20,30\n105,5,-999,Revolving loans,Approved,300,-1,-1\n'}
        for n,s in self.contents.items():
            (self.raw/n).write_text(s, encoding='utf-8')
        self.source = RawDataSource(self.raw)
        self.runtime_path = self.root / 'runtime.json'
        self.runtime_path.write_text(json.dumps({'schema_version':1, 'raw_dir':str(self.raw.relative_to(ROOT)),
            'manifest':str((self.root/'manifest.json').relative_to(ROOT)), 'output_dir':str(self.reports.relative_to(ROOT)),
            'tables':[TRAIN,BUREAU,PREV]}), encoding='utf-8')
        self.runtime = load_config(ROOT, self.runtime_path)
        self.norm_path = self.root/'normalization.json'
        values = json.loads((ROOT/'config/normalization.json').read_text(encoding='utf-8'))
        values['artifact_dir'] = str((self.root/'normalized').relative_to(ROOT))
        values['batch_size'] = 2
        for n in values['tables']:
            by_name = {c['name']:c for c in values['tables'][n]}
            values['tables'][n] = [by_name[k] for k in self.contents[n].splitlines()[0].split(',')]
        self.norm_path.write_text(json.dumps(values), encoding='utf-8')
        self.norm = load_normalization_config(ROOT,self.norm_path)
        self.upstream = self.normalize('source')
        self.history_path = self.root/'history.json'
        self.values = json.loads((ROOT/'config/history_features.json').read_text(encoding='utf-8'))
        self.values.update(artifact_dir=str((self.root/'history').relative_to(ROOT)),case_ids=[1],row_group_size=2048)
        self.config = self.save_config()

    def normalize(self, run_id):
        record = RunRecord(self.runtime, scope='synthetic_full_normalization', metadata={
            'module':'E03','normalization_config_sha256':self.norm.sha256,'data_root':str(self.raw),'fixture_name':run_id})
        result = normalize_dataset(self.source,self.norm,record.run_id)
        record.finish(True,normalization=result)
        return record.path

    def save_config(self):
        self.history_path.write_text(json.dumps(self.values),encoding='utf-8')
        return load_history_config(ROOT,self.history_path)

    def build(self, run_id='first', **kwargs):
        return build_history_features(self.config,self.upstream,self.source,run_id,fixture=True,**kwargs)

    def save_report(self, result):
        scope = 'synthetic_full_history_features' if result['quality']['application_limit'] is None else 'synthetic_prefix_history_features'
        record = RunRecord(self.runtime,scope=scope,metadata={'module':'E04','history_config_sha256':self.config.sha256})
        # 输出包运行编号必须匹配实际生产它的 run_id。
        bundle = json.loads((Path(result['artifact_dir'])/'bundle.json').read_text(encoding='utf-8'))
        record.data['run_id']=bundle['run_id']
        record.finish(True,history_features=result)
        return record.path

    def rows(self,result):
        return {r['SK_ID_CURR']:r for r in pq.read_table(Path(result['artifact_dir'])/'joined_features.parquet').to_pylist()}

    def test_hand_computed_many_to_many_currency_and_label_preservation(self):
        before={n:sha256_file(self.raw/n) for n in self.contents}
        result=self.build()
        rows=self.rows(result)
        self.assertEqual(list(rows),[1,2,3,4])
        self.assertEqual([r['TARGET'] for r in rows.values()],[0,1,0,1])
        r=rows[1]
        self.assertEqual(r['BUREAU_ELIGIBLE_COUNT'],3)
        self.assertEqual(r['PREV_ELIGIBLE_COUNT'],2)
        self.assertEqual(r['BUREAU_C1_CREDIT_SUM'],100)
        self.assertEqual(r['BUREAU_C1_CREDIT_MEAN'],100)
        self.assertEqual(r['BUREAU_C1_AMOUNT_AVAILABLE_COUNT'],1)
        self.assertEqual(r['BUREAU_C1_AMOUNT_UNAVAILABLE_COUNT'],1)
        self.assertEqual(r['BUREAU_OTHER_CURRENCY_COUNT'],1)
        self.assertEqual(r['BUREAU_DAYS_SINCE_MOST_RECENT'],5)
        self.assertAlmostEqual(r['BUREAU_ACTIVE_FRACTION'],1/3)
        self.assertEqual(r['PREV_CASH_COUNT'],1)
        self.assertEqual(r['PREV_CONSUMER_COUNT'],1)
        trace=json.loads((Path(result['artifact_dir'])/'case_trace.json').read_text(encoding='utf-8'))['1']
        self.assertEqual(trace['raw_direct_join_rows'],15)
        self.assertTrue(trace['all_24_features_match'])
        self.assertEqual(before,{n:sha256_file(self.raw/n) for n in self.contents})

    def test_no_history_only_excluded_and_all_missing_amount_are_distinct(self):
        result=self.build()
        rows=self.rows(result)
        self.assertEqual(rows[2]['BUREAU_ELIGIBLE_COUNT'],1)
        self.assertIsNone(rows[2]['BUREAU_C1_CREDIT_SUM'])
        self.assertEqual(rows[2]['BUREAU_C1_AMOUNT_UNAVAILABLE_COUNT'],1)
        for key in (3,4):
            self.assertEqual(rows[key]['BUREAU_ELIGIBLE_COUNT'],0)
            self.assertIsNone(rows[key]['BUREAU_C1_CREDIT_SUM'])
            self.assertIsNone(rows[key]['BUREAU_ACTIVE_FRACTION'])
            self.assertIsNone(rows[key]['PREV_DAYS_SINCE_MOST_RECENT'])
        audit={r['SK_ID_CURR']:r for r in pq.read_table(Path(result['artifact_dir'])/'history_audit.parquet').to_pylist()}
        self.assertEqual(audit[3]['bureau_history_state'],'no_source_history')
        self.assertEqual(audit[4]['bureau_history_state'],'only_time_excluded')
        self.assertEqual({k:rows[3][k] for k in FEATURE_NAMES},{k:rows[4][k] for k in FEATURE_NAMES})

    def test_future_or_day_zero_additions_never_change_candidate_features(self):
        first=self.build('before')
        with (self.raw/BUREAU).open('a',encoding='utf-8') as f:
            f.write('1,1000,-1,1,currency 1,Active,99999999\n3,1001,0,-1,currency 1,Active,99999999\n')
        with (self.raw/PREV).open('a',encoding='utf-8') as f:
            f.write('1000,1,1,Cash loans,Approved,9999999,1000,1000\n1001,3,0,Cash loans,Approved,9999999,1000,1000\n')
        self.upstream=self.normalize('changed-source')
        second=self.build('after')
        a,b=self.rows(first),self.rows(second)
        for key in a:
            self.assertEqual({n:a[key][n] for n in FEATURE_NAMES},{n:b[key][n] for n in FEATURE_NAMES})
        self.assertEqual(first['files']['joined_features.parquet']['sha256'],second['files']['joined_features.parquet']['sha256'])
        self.assertNotEqual(first['quality']['time_audit'],second['quality']['time_audit'])

    def test_held_status_amount_and_future_due_fields_do_not_enter_features(self):
        a=self.build('before')
        p=self.raw/PREV
        p.write_text(self.contents[PREV].replace('Approved','Cancelled').replace('365243','999999').replace('Refused,999,','Approved,777777,'),encoding='utf-8')
        self.upstream=self.normalize('held-changed')
        b=self.build('after')
        self.assertEqual(a['files']['joined_features.parquet']['sha256'],b['files']['joined_features.parquet']['sha256'])
        self.assertNotIn('TARGET',feature_contract()['feature_columns'])
        self.assertFalse(any('audit' in n.lower() for n in FEATURE_NAMES))

    def test_unlabeled_history_retained_in_summaries_but_not_added_as_labeled_rows(self):
        result=self.build()
        bureau=pq.read_table(Path(result['artifact_dir'])/'bureau_features.parquet').to_pylist()
        self.assertIn(5,[r['SK_ID_CURR'] for r in bureau])
        self.assertEqual(len(self.rows(result)),4)

    def test_empty_eligible_sets_and_missing_time_are_supported(self):
        b=self.raw/BUREAU
        p=self.raw/PREV
        b.write_text(self.contents[BUREAU].splitlines()[0]+'\n1,10,,-1,currency 1,Active,100\n2,11,-1,,currency 1,Active,100\n',encoding='utf-8')
        p.write_text(self.contents[PREV].splitlines()[0]+'\n100,1,,Cash loans,Approved,100,365243,365243\n',encoding='utf-8')
        self.upstream=self.normalize('no-eligible')
        r=self.build()
        self.assertEqual(r['quality']['time_audit']['bureau']['eligible_rows'],0)
        self.assertEqual(r['quality']['time_audit']['previous']['eligible_rows'],0)
        for row in self.rows(r).values():
            self.assertEqual(row['BUREAU_ELIGIBLE_COUNT'],0)
            self.assertEqual(row['PREV_ELIGIBLE_COUNT'],0)
            self.assertIsNone(row['BUREAU_C1_CREDIT_SUM'])

    def test_zero_credit_amount_stays_observed_zero(self):
        (self.raw/BUREAU).write_text(self.contents[BUREAU].replace('currency 1,Active,100\n','currency 1,Active,0\n'),encoding='utf-8')
        self.upstream=self.normalize('zero-amount')
        r=self.rows(self.build())[1]
        self.assertEqual(r['BUREAU_C1_CREDIT_SUM'],0)
        self.assertEqual(r['BUREAU_C1_AMOUNT_AVAILABLE_COUNT'],1)
        self.assertEqual(r['BUREAU_C1_AMOUNT_UNAVAILABLE_COUNT'],1)

    def test_missing_required_history_field_is_rejected(self):
        lines=[r.split(',') for r in self.contents[BUREAU].splitlines()]
        index=lines[0].index('CREDIT_CURRENCY')
        for r in lines:
            del r[index]
        (self.raw/BUREAU).write_text('\n'.join(','.join(r) for r in lines)+'\n',encoding='utf-8')
        values=json.loads(self.norm_path.read_text(encoding='utf-8'))
        values['tables'][BUREAU]=[c for c in values['tables'][BUREAU] if c['name']!='CREDIT_CURRENCY']
        self.norm_path.write_text(json.dumps(values),encoding='utf-8')
        self.norm=load_normalization_config(ROOT,self.norm_path)
        self.upstream=self.normalize('missing-column')
        with self.assertRaises(DataError) as caught:
            self.build()
        self.assertEqual(caught.exception.code,'history_columns_missing')

    def test_dataset_promotion_failure_removes_new_complete_staging(self):
        original=Path.replace
        def fail_output(path,target):
            if path.is_dir() and path.name.endswith('.partial') and path.is_relative_to(self.config.artifact_dir):
                raise OSError('simulated directory publication failure')
            return original(path,target)
        with patch('credit_risk.history.Path.replace',new=fail_output):
            with self.assertRaises(OSError):
                self.build()
        self.assertFalse(list(self.config.artifact_dir.rglob('*.parquet')))
        self.assertFalse(list(self.config.artifact_dir.rglob('*.partial')))

    def test_prefix_uses_all_history_of_selected_applications(self):
        r=self.build(application_limit=1)
        self.assertEqual(list(self.rows(r)),[1])
        self.assertEqual(self.rows(r)[1]['BUREAU_ELIGIBLE_COUNT'],3)
        self.assertEqual(r['quality']['time_audit']['bureau']['raw_rows_in_scope'],5)
        with self.assertRaises(DataError):
            load_history_bundle(ROOT,self.save_report(r),require_official=False)

    def test_repeat_is_deterministic_not_append_and_existing_directory_is_protected(self):
        a,b=self.build('a'),self.build('b')
        for name in ('bureau_features.parquet','previous_features.parquet','history_audit.parquet','joined_features.parquet'):
            self.assertEqual(a['files'][name]['sha256'],b['files'][name]['sha256'])
        with self.assertRaises(DataError) as caught:
            self.build('a')
        self.assertEqual(caught.exception.code,'history_dataset_exists')

    def test_duplicate_summary_key_stops_before_join(self):
        sql=copy.deepcopy(self.config.sql)
        first=next(iter(sql))
        sql[first]['text']+='\nINSERT INTO bureau_features SELECT * FROM bureau_features LIMIT 1;'
        self.config=replace(self.config,sql=sql)
        with self.assertRaises(DataError) as caught:
            self.build()
        self.assertEqual(caught.exception.code,'history_duplicate_key')
        self.assertFalse(list(self.config.artifact_dir.rglob('*.parquet')))

    def test_mid_batch_failure_preserves_previous_complete_dataset(self):
        first=self.build('old')
        old=sha256_file(Path(first['artifact_dir'])/'bundle.json')
        from credit_risk.history import copy_verified as original
        calls=[]
        def fail_second(*args,**kwargs):
            calls.append(1)
            if len(calls)==2:
                raise OSError('simulated write interruption')
            return original(*args,**kwargs)
        with patch('credit_risk.history.copy_verified',side_effect=fail_second):
            with self.assertRaises(OSError):
                self.build('new')
        self.assertEqual(old,sha256_file(Path(first['artifact_dir'])/'bundle.json'))
        self.assertFalse((self.config.artifact_dir/'full/new').exists())
        self.assertFalse(list(self.config.artifact_dir.rglob('*.partial')))

    def test_manual_case_mismatch_stops_publication(self):
        from credit_risk.history import manual_features as original
        def wrong(*args):
            r=original(*args)
            r['BUREAU_C1_CREDIT_SUM']=12345
            return r
        with patch('credit_risk.history.manual_features',side_effect=wrong):
            with self.assertRaises(DataError) as caught:
                self.build()
        self.assertEqual(caught.exception.code,'history_case_mismatch')
        self.assertFalse(list(self.config.artifact_dir.rglob('*.parquet')))

    def test_raw_snapshot_mismatch_and_missing_case_reject(self):
        self.values['case_ids']=[99]
        self.config=self.save_config()
        with self.assertRaises(DataError) as caught:
            self.build('missing-case')
        self.assertEqual(caught.exception.code,'history_case_missing')
        (self.raw/BUREAU).write_text(self.contents[BUREAU]+'1,999,-1,-1,currency 1,Active,10\n',encoding='utf-8')
        with self.assertRaises(DataError) as caught:
            self.build('changed')
        self.assertEqual(caught.exception.code,'history_raw_changed')

    def test_incomplete_upstream_and_wrong_official_scope_are_rejected(self):
        with self.assertRaises(DataError):
            build_history_features(self.config,self.upstream,self.source,'official',fixture=False)
        report=json.loads(self.upstream.read_text(encoding='utf-8'))
        report['status']='failed'
        self.upstream.write_text(json.dumps(report),encoding='utf-8')
        with self.assertRaises(DataError):
            self.build()

    def test_downstream_allowlist_and_tampered_artifact(self):
        result=self.build()
        report=self.save_report(result)
        loaded=load_history_bundle(ROOT,report,require_official=False)
        self.assertEqual(loaded['history_feature_columns'],FEATURE_NAMES)
        self.assertEqual(len(FEATURE_NAMES),24)
        self.assertNotIn('TARGET',loaded['history_feature_columns'])
        with self.assertRaises(DataError):
            load_history_bundle(ROOT,report)
        (loaded['directory']/'quality.json').write_text('{}',encoding='utf-8')
        with self.assertRaises(DataError):
            load_history_bundle(ROOT,report,require_official=False)

    def test_invalid_config_paths_policy_limits_and_ids(self):
        valid=copy.deepcopy(self.values)
        for key,value in [('artifact_dir','data/raw/bad'),('artifact_dir','../escape'),('duckdb_threads',True),
                          ('case_ids',[]),('case_ids',[1,1]),('case_ids',[True]),('policy_version','allow_future'),
                          ('row_group_size',1),('manual_absolute_tolerance',1),('sql_files',['../bad.sql'])]:
            self.values={**valid,key:value}
            with self.assertRaises(DataError):
                self.save_config()
        for limit in (0,-1,True,100001):
            with self.assertRaises(DataError):
                self.build(application_limit=limit)
        with self.assertRaises(DataError):
            self.build('../unsafe')

    def test_cli_complete_chain_and_failure_record_from_another_directory(self):
        before=sha256_file(ROOT/'config/experiment_plan.json')
        args=[sys.executable,'-X','utf8',str(ROOT/'scripts/run_history_features.py'),'--input-run',str(self.upstream),
              '--config',str(self.runtime_path),'--history-config',str(self.history_path),'--fixture']
        result=subprocess.run(args,cwd=self.root,capture_output=True,text=True,encoding='utf-8',timeout=40)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        out=json.loads(result.stdout)
        loaded=load_history_bundle(ROOT,out['report'],require_official=False)
        self.assertTrue(loaded['joined_features'].exists())
        self.assertEqual(before,sha256_file(ROOT/'config/experiment_plan.json'))
        result=subprocess.run([*args,'--data-dir',str(ROOT/'data/raw')],cwd=self.root,capture_output=True,text=True,encoding='utf-8',timeout=20)
        self.assertEqual(result.returncode,2)
        out=json.loads(result.stdout)
        report=json.loads(Path(out['report']).read_text(encoding='utf-8'))
        self.assertEqual(report['status'],'failed')
        self.assertEqual(report['error']['code'],'fixture_path_invalid')


if __name__=='__main__':
    unittest.main()
