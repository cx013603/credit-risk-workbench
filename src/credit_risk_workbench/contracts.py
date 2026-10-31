"""V2 public schema and white-listed projections; stdlib only for frozen worker."""
import hashlib
import json
import math

SCHEMA='workbench_api_v2.0'
METHOD='training_count_weighted_path_log_odds'
MAX_BATCH=20
MAX_PAGE=50
class ContractError(ValueError):pass

def require(value,code='INVALID_INPUT'):
    if not value:raise ContractError(code)

def integer(value,minimum=1,maximum=2**63-1):
    require(type(value) is int and minimum<=value<=maximum);return value

def population(value):
    require(value in ['labeled','unlabeled'],'INVALID_POPULATION');return value

def identifiers(values):
    require(isinstance(values,list) and 1<=len(values)<=MAX_BATCH)
    for value in values:integer(value)
    require(len(set(values))==len(values));return values

def revision(seal,dictionary_hash,pop,ident):
    return hashlib.sha256(f'{SCHEMA}|{seal}|{dictionary_hash}|{pop}|{ident}'.encode()).hexdigest()

def project_case(case,dictionary,seal,dictionary_hash):
    r=case['result'];terms=case['terms'];fields=dictionary['fields'];names=dictionary['order']
    require([t['feature'] for t in terms]==names and len(names)==40,'VERSION_MISMATCH')
    require(r['feature_version']=='application_contract_v1' and r['tree_version']=='tree_v1' and case['method']==METHOD,'VERSION_MISMATCH')
    require(abs(case['base_log_odds']+math.fsum(t['contribution_log_odds'] for t in terms)-case['raw_score'])<=1e-10,'VERSION_MISMATCH')
    require(abs(1/(1+math.exp(-case['raw_score']))-r['risk_probability'])<=1e-10,'VERSION_MISMATCH')
    values={t['feature']:t['value'] for t in terms};facts=[]
    for t,m in zip(terms,fields):
        require(t['feature']==m['feature'],'VERSION_MISMATCH')
        no_history=m['group'] in ['bureau','previous'] and values['BUREAU_ELIGIBLE_COUNT' if m['group']=='bureau' else 'PREV_ELIGIBLE_COUNT']==0
        facts.append({'fact_id':t['feature'],'label':m['label'],'raw_value':t['value'],'unit':m['unit'],
            'value_kind':'source_field' if m['group']=='application' else 'aggregate',
            'missing_state':'no_eligible_history' if no_history else 'source_null' if t['value'] is None else 'observed',
            'source':{'table':('application_train.csv' if case['population']=='labeled' else 'application_test.csv') if m['group']=='application' else m['source'],'record_key_kind':m['source_key'],'detail_status':'source_not_provided'},
            'rule':{'formula':m['formula'],'time_rule':m['time_rule'],'version':'history_before_day_zero_currency_separate_v1' if m['group']!='application' else 'application_contract_v1'},
            'encoding_category_status':'not_inspected' if m['type']=='string' else 'not_applicable'})
    models={'recommended':'tree','tree':{'version':r['tree_version'],'run_id':r['tree_run_id'],'probability':r['risk_probability'],'threshold':r['threshold'],'risk_flag':r['risk_flag'],'threshold_scope':'original_validation_locked_classroom_threshold'},
        'logistic':{'version':r['logistic_version'],'run_id':r['logistic_run_id'],'probability':r['logistic_probability']},
        'scorecard':{'version':r['scorecard_version'],'run_id':r['scorecard_run_id'],'probability':r['scorecard_probability'],'score':r['scorecard_score'],'score_unit':'scorecard_points'}}
    versions={k:r[k] for k in ['dataset_version','feature_version','feature_contract_sha256','split_manifest_sha256','tree_version','tree_run_id','logistic_version','scorecard_version','assessment_run_id','scoring_version']}
    versions.update(release_manifest_sha256=seal,display_dictionary_sha256=dictionary_hash,explanation_method=METHOD)
    result={'case_key':{'population':case['population'],'application_id':r['SK_ID_CURR']},'case_revision':revision(seal,dictionary_hash,case['population'],r['SK_ID_CURR']),
        'sample_use':r['sample_use'],'versions':versions,
        'profile':{'scope':'saved_40_model_inputs_only','facts':facts,'additional_application_fields_status':'not_connected'},
        'model_outputs':models,'explanation':{'method':METHOD,'unit':'log_odds','base_log_odds':case['base_log_odds'],'raw_score':case['raw_score'],'contribution_sum':case['contribution_sum'],
            'terms':[{'fact_id':t['feature'],'contribution_log_odds':t['contribution_log_odds']} for t in terms], 'feature_count':40},
        'availability':{'history_details':'source_not_provided','train_reference':'source_not_provided','controlled_narrative':'not_connected','review':'not_connected'},
        'limits':case['limits']+[case['sample_notice'],'Source amount currency/period unverified; no fixed-horizon PD or bank approval.']}
    json.dumps(result,allow_nan=False);return result
