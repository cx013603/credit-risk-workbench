import copy,json,math,platform,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import duckdb,pyarrow as pa,pyarrow.parquet as pq
from credit_risk.raw import DataError,sha256_file
from credit_risk_workbench.features_v2 import VERSION,feature_definitions,run_family,join_new,digest,write,verify_frozen,load_partition
from credit_risk_workbench.feature_reference_v2 import reference_features

class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.db=duckdb.connect();self.db.execute('CREATE TABLE allowed(SK_ID_CURR BIGINT);INSERT INTO allowed VALUES(1),(2),(3)')
        self.data={
         'previous_raw':[{'SK_ID_CURR':1,'SK_ID_PREV':n,'DAYS_DECISION':-50} for n in [11,12,13]]+[{'SK_ID_CURR':2,'SK_ID_PREV':21,'DAYS_DECISION':0}],
         'bureau_raw':[{'SK_ID_CURR':1,'SK_ID_BUREAU':101,'DAYS_CREDIT':-100,'DAYS_CREDIT_UPDATE':-1},{'SK_ID_CURR':1,'SK_ID_BUREAU':102,'DAYS_CREDIT':-100,'DAYS_CREDIT_UPDATE':0}],
         'cc_input':[{'SK_ID_CURR':1,'SK_ID_PREV':11,'MONTHS_BALANCE':-1,'SK_DPD':0},{'SK_ID_CURR':1,'SK_ID_PREV':11,'MONTHS_BALANCE':-2,'SK_DPD':8},{'SK_ID_CURR':1,'SK_ID_PREV':12,'MONTHS_BALANCE':-13,'SK_DPD':20}],
         'pos_input':[{'SK_ID_CURR':1,'SK_ID_PREV':11,'MONTHS_BALANCE':-1,'SK_DPD':None},{'SK_ID_CURR':1,'SK_ID_PREV':11,'MONTHS_BALANCE':-2,'SK_DPD':-3},{'SK_ID_CURR':1,'SK_ID_PREV':11,'MONTHS_BALANCE':-5,'SK_DPD':0}],
         'bb_input':[{'SK_ID_BUREAU':101,'MONTHS_BALANCE':-i,'STATUS':v} for i,v in enumerate(['X','C','0','2'],1)]+[{'SK_ID_BUREAU':102,'MONTHS_BALANCE':-1,'STATUS':'5'}],
         'inst_input':[]}
        for loan,version,number,due,paid,amount in [(11,1,1,-20,-25,30),(11,1,1,-20,-18,70),(11,1,2,-10,-12,100),(12,2,1,-8,-7,100),(12,3,2,-3,-2,100),(13,0,1,-2,-1,100)]:
            self.data['inst_input'].append(dict(SK_ID_CURR=1,SK_ID_PREV=loan,NUM_INSTALMENT_VERSION=version,NUM_INSTALMENT_NUMBER=number,DAYS_INSTALMENT=due,DAYS_ENTRY_PAYMENT=paid,AMT_INSTALMENT=100,AMT_PAYMENT=amount))
    def tearDown(self):self.db.close()
    def run_all(self):
        schemas={'previous_raw':'SK_ID_CURR BIGINT,SK_ID_PREV BIGINT,DAYS_DECISION DOUBLE','bureau_raw':'SK_ID_CURR BIGINT,SK_ID_BUREAU BIGINT,DAYS_CREDIT DOUBLE,DAYS_CREDIT_UPDATE DOUBLE','cc_input':'SK_ID_CURR BIGINT,SK_ID_PREV BIGINT,MONTHS_BALANCE BIGINT,SK_DPD DOUBLE','pos_input':'SK_ID_CURR BIGINT,SK_ID_PREV BIGINT,MONTHS_BALANCE BIGINT,SK_DPD DOUBLE','bb_input':'SK_ID_BUREAU BIGINT,MONTHS_BALANCE BIGINT,STATUS VARCHAR','inst_input':'SK_ID_CURR BIGINT,SK_ID_PREV BIGINT,NUM_INSTALMENT_VERSION DOUBLE,NUM_INSTALMENT_NUMBER BIGINT,DAYS_INSTALMENT DOUBLE,DAYS_ENTRY_PAYMENT DOUBLE,AMT_INSTALMENT DOUBLE,AMT_PAYMENT DOUBLE'}
        for name,schema in schemas.items():
            self.db.execute('CREATE TABLE '+name+'('+schema+')');cols=[s.strip().split()[0] for s in schema.split(',')]
            if self.data[name]:self.db.executemany('INSERT INTO '+name+' VALUES('+','.join('?' for _ in cols)+')',[[r[c] for c in cols] for r in self.data[name]])
        for family in ['CC','POS','BB','INST']:run_family(self.db,ROOT,family)
        join_new(self.db);cur=self.db.execute('SELECT * FROM new_features ORDER BY SK_ID_CURR');names=[c[0] for c in cur.description];return {r[0]:dict(zip(names[1:],r[1:])) for r in cur.fetchall()}
    def test_all_31_SQL_features_match_independent_Python(self):
        actual=self.run_all();expected=reference_features([1,2,3],self.data['previous_raw'],self.data['bureau_raw'],cc=self.data['cc_input'],pos=self.data['pos_input'],bb=self.data['bb_input'],inst=self.data['inst_input'])
        self.assertEqual(set(actual),set(expected))
        for i in actual:
            for name,v in actual[i].items():
                if v is None:self.assertIsNone(expected[i][name],(i,name))
                else:self.assertAlmostEqual(v,expected[i][name],places=12,msg=name)
    def test_weighted_month_ratio_not_mean_of_loan_rates(self):self.assertAlmostEqual(self.run_all()[1]['CC_POSITIVE_DPD_FRACTION'],2/3)
    def test_unknown_DPD_is_not_zero(self):
        r=self.run_all()[1];self.assertEqual((r['POS_DPD_UNKNOWN_COUNT'],r['POS_MAX_DPD'],r['POS_POSITIVE_DPD_FRACTION']),(2,0,0))
    def test_unknown_and_closed_bureau_status_not_active_zero(self):
        r=self.run_all()[1];self.assertEqual((r['BB_UNKNOWN_STATUS_FRACTION'],r['BB_CLOSED_STATUS_FRACTION'],r['BB_DELINQUENT_STATUS_FRACTION']),(.25,.25,.5))
    def test_multiple_payments_and_versions_are_separate(self):
        r=self.run_all()[1];self.assertEqual((r['INST_PAYMENT_RECORD_COUNT'],r['INST_EXCLUDED_CALENDAR_LOAN_COUNT'],r['INST_SINGLE_CALENDAR_GROUP_COUNT']),(6,2,2));self.assertEqual((r['INST_LATE_PAYMENT_GROUP_FRACTION'],r['INST_MULTI_PAYMENT_GROUP_FRACTION'],r['INST_MAX_PAYMENT_DELAY_DAYS']),(.5,.5,2))
    def test_no_history_counts_zero_rates_recent_null(self):
        r=self.run_all()[3]
        for f in feature_definitions():self.assertEqual(r[f['name']],0 if f['kind']=='count' else None)
    def test_future_calendar_does_not_change_existing_payments(self):
        baseline=copy.deepcopy(self.data);extra=copy.deepcopy(self.data['inst_input'][0]);extra.update(NUM_INSTALMENT_VERSION=99,DAYS_ENTRY_PAYMENT=1);self.data['inst_input'].append(extra);r=self.run_all();expected=reference_features([1],baseline['previous_raw'],baseline['bureau_raw'],cc=baseline['cc_input'],pos=baseline['pos_input'],bb=baseline['bb_input'],inst=baseline['inst_input']);self.assertEqual(r[1],expected[1])
    def test_unknown_future_zero_and_wrong_pair_excluded(self):
        base=copy.deepcopy(self.data)
        for month in [0,1,None]:self.data['cc_input'].append(dict(SK_ID_CURR=1,SK_ID_PREV=11,MONTHS_BALANCE=month,SK_DPD=999))
        self.data['cc_input'].extend([dict(SK_ID_CURR=2,SK_ID_PREV=11,MONTHS_BALANCE=-1,SK_DPD=999),dict(SK_ID_CURR=2,SK_ID_PREV=21,MONTHS_BALANCE=-1,SK_DPD=999)])
        for day in [None,0,1]:
            r=copy.deepcopy(self.data['inst_input'][0]);r['DAYS_ENTRY_PAYMENT']=day;self.data['inst_input'].append(r)
        actual=self.run_all();expected=reference_features([1,2,3],base['previous_raw'],base['bureau_raw'],cc=base['cc_input'],pos=base['pos_input'],bb=base['bb_input'],inst=base['inst_input']);self.assertEqual(actual,expected)
    def test_duplicate_loan_month_rejected(self):
        self.data['cc_input'].append(copy.deepcopy(self.data['cc_input'][0]))
        with self.assertRaises(DataError):self.run_all()
    def test_duplicate_exact_payment_rejected(self):
        self.data['inst_input'].append(copy.deepcopy(self.data['inst_input'][0]))
        with self.assertRaises(DataError):self.run_all()
    def test_conflicting_prescribed_amount_rejected(self):
        self.data['inst_input'][1]['AMT_INSTALMENT']=999
        with self.assertRaises(DataError):self.run_all()

class FrozenInputTests(unittest.TestCase):
    def setUp(self):
        self.fixture_parent=ROOT/'reports/workbench/feature_v2_test_fixtures';self.fixture_parent.mkdir(exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=self.fixture_parent);self.root=Path(self.tmp.name).resolve();self.folder=self.root/'package';self.folder.mkdir();self.bound=self.root/'policy.json';self.bound.write_text('{}')
        names=['F'+str(i) for i in range(71)];write(self.folder/'contract.json',{'version':VERSION,'feature_columns':names});protocol={'purpose':'fixture'};write(self.folder/'protocol.json',protocol)
        for partition,ids,folds in [('train',[1,2,3],[1,2,3]),('validation',[4],[None]),('unlabeled_scoring',[5],[None])]:pq.write_table(pa.table({'SK_ID_CURR':ids,'sample_use':[partition]*len(ids),'CV_FOLD':folds,**{n:[float(i) for i in ids] for n in names}}),self.folder/(partition+'.parquet'))
        files={p.name:{'sha256':sha256_file(p)} for p in self.folder.iterdir()};write(self.folder/'manifest.json',{'files':files,'bindings':{'policy.json':sha256_file(self.bound)},'environment':{'python':platform.python_version()}})
        write(self.folder/'freeze.json',{'version':VERSION,'G3_passed':True,'training_started':False,'manifest_sha256':sha256_file(self.folder/'manifest.json'),'protocol_fingerprint':digest(protocol)});self.seal=sha256_file(self.folder/'freeze.json')
    def tearDown(self):
        self.assertTrue(self.root.is_relative_to(self.fixture_parent.resolve()));self.tmp.cleanup()
    def test_fit_view_contains_only_complement_training_features(self):
        r=load_partition(self.root,self.folder,self.seal,'train',purpose='fit',heldout_fold=2);self.assertEqual(r['identity']['SK_ID_CURR'].to_pylist(),[1,3]);self.assertEqual(len(r['X'].column_names),71);self.assertFalse(set(r['X'].column_names)&{'TARGET','SK_ID_CURR','CV_FOLD','sample_use'})
    def test_validation_and_unlabeled_fit_rejected(self):
        for p in ['validation','unlabeled_scoring']:
            with self.assertRaises(DataError):load_partition(self.root,self.folder,self.seal,p,purpose='fit')
    def test_reserved_partition_rejected_before_file_read(self):
        with self.assertRaises(DataError):load_partition(self.root,self.folder,self.seal,'test')
    def test_changed_bound_policy_rejected(self):
        self.bound.write_text('{"changed":true}')
        with self.assertRaises(DataError):verify_frozen(self.root,self.folder,self.seal)
    def test_changed_feature_bytes_rejected(self):
        (self.folder/'train.parquet').write_bytes(b'broken')
        with self.assertRaises(DataError):load_partition(self.root,self.folder,self.seal,'train',purpose='fit')
    def test_missing_or_changed_freeze_blocks_input(self):
        with self.assertRaises(DataError):verify_frozen(self.root,self.folder,'a'*64)
    def test_invalid_fold_rejected(self):
        for fold in [0,6,True,'1']:
            with self.assertRaises(DataError):load_partition(self.root,self.folder,self.seal,'train',purpose='fit',heldout_fold=fold)
    def test_validation_score_allowed_without_fit(self):
        identity=load_partition(self.root,self.folder,self.seal,'validation')['identity']
        self.assertEqual(identity['SK_ID_CURR'].to_pylist(),[4]);self.assertEqual(identity['CV_FOLD'].to_pylist(),[None])
if __name__=='__main__':unittest.main()
