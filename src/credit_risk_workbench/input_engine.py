"""Raw inputs -> original frozen E04 SQL -> saved original scoring processors."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import duckdb
import pyarrow as pa
from credit_risk.history import SQL_FILES,join_query
from credit_risk.scoring import validate_features
from .contracts import require
from .input_contract import MAIN,BUREAU,PREVIOUS,VERSION,SEAL,digest,require_complete

# Compute extension identity before frozen_guard installs its original-directory deny.
ADAPTER_HASHES={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in ['input_engine.py','input_contract.py']}

class InputEngine:
    def __init__(self,release,seal,contract):
        require(seal==SEAL,'VERSION_MISMATCH')
        self.release=Path(release);self.contract=contract
        manifest=json.loads((self.release/'portable_manifest.json').read_text(encoding='utf-8'))
        entries=manifest['files']
        # SQL bytes are checked against the externally sealed release manifest.
        self.sql={}
        for name in SQL_FILES:
            p=self.release/'project'/name;raw=p.read_bytes();key='project/'+name
            entry=entries[key]
            require(hashlib.sha256(raw).hexdigest()==entry['sha256'],'VERSION_MISMATCH')
            self.sql[name]={'text':raw.decode('utf-8'),'sha256':entry['sha256']}
        require(contract['main_features']==MAIN and len(contract['feature_columns'])==40,'VERSION_MISMATCH')
        self.lineage={'input_contract':VERSION,'feature_version':contract['version'],'release_manifest_sha256':seal,'input_adapter_sha256':ADAPTER_HASHES,'sql_sha256':{p:v['sha256'] for p,v in self.sql.items()}}
    def build(self,payloads):
        require(isinstance(payloads,list) and 1<=len(payloads)<=5,'INVALID_INPUT')
        for p in payloads:require_complete(p)
        ids=[p['application']['SK_ID_CURR'] for p in payloads];require(len(set(ids))==len(ids),'INVALID_INPUT')
        types={f['name']:getattr(pa,f['type'])() for f in self.contract['features']}
        main_schema=pa.schema([pa.field('SK_ID_CURR',pa.int64()),*[pa.field(n,types[n]) for n in MAIN],pa.field('file_row_number',pa.int64())])
        schemas={
            'bureau':pa.schema([pa.field(n,pa.int64() if n in ['SK_ID_CURR','SK_ID_BUREAU'] else pa.float64() if n in ['DAYS_CREDIT','DAYS_CREDIT_UPDATE','AMT_CREDIT_SUM'] else pa.string()) for n in BUREAU]),
            'previous':pa.schema([pa.field(n,pa.int64() if n in ['SK_ID_CURR','SK_ID_PREV'] else pa.float64() if n=='DAYS_DECISION' else pa.string()) for n in PREVIOUS])}
        with duckdb.connect(':memory:') as db:
            db.execute("SET memory_limit='128MB'");db.execute('SET threads=1');db.execute("SET max_temp_directory_size='0B'")
            db.register('application_input',pa.Table.from_pylist([{**p['application'],'file_row_number':i} for i,p in enumerate(payloads)],schema=main_schema))
            for family in ['bureau','previous']:
                rows=[r for p in payloads for r in p[family]['rows']]
                key='SK_ID_BUREAU' if family=='bureau' else 'SK_ID_PREV'
                require(len({r[key] for r in rows})==len(rows),'INPUT_DUPLICATE_HISTORY_KEY')
                db.register(family+'_input',pa.Table.from_pylist(rows,schema=schemas[family]))
            db.execute(self.sql[SQL_FILES[0]]['text']);db.execute(self.sql[SQL_FILES[1]]['text'])
            table=db.sql(join_query(SimpleNamespace(sql=self.sql))).to_arrow_table().select(self.contract['feature_columns'])
        table=table.cast(pa.schema([pa.field(n,types[n]) for n in self.contract['feature_columns']]))
        require(table.num_rows==len(payloads),'VERSION_MISMATCH');validate_features(table,self.contract)
        return table
    def score(self,loaded,payloads):
        require(isinstance(payloads,list) and 1<=len(payloads)<=5,'INVALID_INPUT')
        for p in payloads:
            require_complete(p)
            s=loaded.sources['labeled'];ident=p['application']['SK_ID_CURR']
            require(ident not in s.index or s.features['sample_use'][s.index[ident]].as_py()!='test','SAMPLE_USE_NOT_ALLOWED')
        table=self.build(payloads);source=loaded.sources['labeled']
        require(source.contract==self.contract,'VERSION_MISMATCH')
        values,_=source._score_features(table)
        explanation=loaded.explanation.explainer.explain(table)
        results=[]
        for j,p in enumerate(payloads):
            probability=float(values['risk_probability'][j]);require(abs(probability-float(explanation['probability'][j]))<=1e-10,'VERSION_MISMATCH')
            features=table.slice(j,1).to_pylist()[0]
            states={}
            for name,value in features.items():
                prefix='BUREAU' if name.startswith('BUREAU_') else 'PREV' if name.startswith('PREV_') else None
                states[name]='no_eligible_history' if prefix and features[prefix+'_ELIGIBLE_COUNT']==0 else 'no_eligible_amount' if value is None and name in ['BUREAU_C1_CREDIT_SUM','BUREAU_C1_CREDIT_MEAN','BUREAU_C1_CREDIT_MAX'] else 'source_null' if value is None else 'observed'
            results.append({'input_sha256':digest(p),'input_contract_version':VERSION,'case_key':{'population':'new_application','application_id':p['application']['SK_ID_CURR']},'sample_use':'new_input_prediction_only','versions':source.bundle['output_versions'],**self.lineage,'threshold':source.bundle['threshold'],'probability':probability,'risk_flag':bool(values['risk_flag'][j]),'scorecard_probability':float(values['scorecard_probability'][j]),'scorecard_score':float(values['scorecard_score'][j]),'logistic_probability':float(values['logistic_probability'][j]),'features':features,'feature_states':states,'feature_sha256':digest(features),'explanation':{'method':'training_count_weighted_path_log_odds','base_log_odds':float(loaded.explanation.explainer.base),'raw_score':float(explanation['raw_score'][j]),'terms':[{'feature':n,'value':features[n],'contribution_log_odds':float(explanation['contributions'][j,k])} for k,n in enumerate(self.contract['feature_columns'])],'checks':explanation['checks']},'validation':require_complete(p),'source_kind':p['source_kind'],'source_reference':p['source_reference'],'limits':['原40项保存模型；新71项候选未启用。','无真实标签，仅预测；不新增AUC或独立测试结论。','匿名来源金额尺度不等于人民币或月薪。','原始资料来源声明由提交者提供，系统不能证明真实申请前可获得性。','路径贡献不是因果解释或真实银行审批理由。']})
        return {'results':results,'count':len(results),'model_activation':'original40_only'}
