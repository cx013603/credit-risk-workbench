"""Deterministic Chinese explanations over the existing40 facts and original tree terms."""
import hashlib
import json
import math
from pathlib import Path

from .contracts import METHOD
from .history_index import digest

VERSION='workbench_narratives_zh_v1'
METHOD_ZH='deterministic_fact_model_review_v1'


class NarrativeError(ValueError):pass


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,allow_nan=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def formatted(value):
    if value is None:return 'null'
    if isinstance(value,str):return json.dumps(value,ensure_ascii=False)
    return format(value,',.10g')


class NarrativeEngine:
    def __init__(self,path,dictionary_path,dictionary_sha256):
        try:
            if digest(dictionary_path)!=dictionary_sha256:raise ValueError('Dictionary binding')
            self.templates=json.loads(Path(path).read_text(encoding='utf-8'))
            dictionary=json.loads(Path(dictionary_path).read_text(encoding='utf-8'))
            t=self.templates
            if t['version']!=VERSION or t['method']!=METHOD_ZH or t['contribution_method']!=METHOD or len(t['fields'])!=40 or t['sections']!=['fact','model_response','review','evidence','limits'] or not t['limits']:raise ValueError('Version/coverage')
            if [f['fact_id'] for f in t['fields']]!=dictionary['order']:raise ValueError('Field order')
            for f,d in zip(t['fields'],dictionary['fields']):
                if any(f[k]!=d[v] for k,v in [('fact_id','feature'),('label','label'),('group','group'),('unit','unit'),('formula','formula'),('source_key','source_key')]):raise ValueError('Metadata binding')
                if any(not isinstance(f.get(k),str) or not f[k].strip() for k in ['definition_zh','review_zh','unit_zh','time_zh','missing_policy','direction_policy','evidence_policy']):raise ValueError('Missing definition')
            self.sha256=digest(path);self.dictionary_sha256=dictionary_sha256
        except (OSError,ValueError,TypeError,KeyError) as exc:raise NarrativeError('NARRATIVE_CONFIG_INVALID') from exc

    def info(self):
        return {'status':'ready','version':VERSION,'method':METHOD_ZH,'template_sha256':self.sha256,'dictionary_sha256':self.dictionary_sha256,'feature_count':40,'generation':'deterministic_templates_no_LLM','sections':self.templates['sections']}

    def render(self,case,history=None):
        try:
            facts=case['profile']['facts'];terms=case['explanation']['terms'];metadata=self.templates['fields']
            if case['sample_use'] not in ['train','validation','unlabeled_scoring'] or case['explanation']['method']!=METHOD or case['versions']['display_dictionary_sha256']!=self.dictionary_sha256:raise ValueError('Case binding')
            if len(facts)!=40 or len(terms)!=40 or [f['fact_id'] for f in facts]!=[m['fact_id'] for m in metadata] or [t['fact_id'] for t in terms]!=[f['fact_id'] for f in facts]:raise ValueError('Fact/term order')
            values={f['fact_id']:f['raw_value'] for f in facts}
            if (case['case_key']['population']=='unlabeled')!=(case['sample_use']=='unlabeled_scoring'):raise ValueError('Population/use')
            for prefix in ['BUREAU','PREV']:
                n=values[prefix+'_ELIGIBLE_COUNT']
                if values[prefix+'_HAS_ELIGIBLE_HISTORY']!=int(n>0):raise ValueError('History presence flag')
            n=values['BUREAU_ELIGIBLE_COUNT']
            if n>0 and (values['BUREAU_ACTIVE_FRACTION'] is None or not math.isclose(values['BUREAU_ACTIVE_FRACTION'],values['BUREAU_ACTIVE_COUNT']/n,abs_tol=1e-12,rel_tol=1e-12)):raise ValueError('Ratio denominator')
            for f,m,t in zip(facts,metadata,terms):
                v=f['raw_value'];contribution=t['contribution_log_odds']
                if f['label']!=m['label'] or f['unit']!=m['unit'] or f['rule']['formula']!=m['formula'] or type(v) is bool or not (v is None or isinstance(v,str) or type(v) in [int,float] and math.isfinite(v)) or type(contribution) not in [int,float] or not math.isfinite(contribution):raise ValueError('Fact type/metadata')
                count=values.get('BUREAU_ELIGIBLE_COUNT' if m['group']=='bureau' else 'PREV_ELIGIBLE_COUNT')
                if m['group']!='application' and (type(count) is not int or count<0):raise ValueError('History count')
                state='no_eligible_history' if m['group']!='application' and count==0 else 'source_null' if v is None else 'observed'
                if f['missing_state']!=state:raise ValueError('Missing semantics')
                if m['group']!='application' and state=='no_eligible_history' and ((m['unit'] in ['records','binary'] and v!=0) or (m['unit'] not in ['records','binary'] and v is not None)):raise ValueError('No-history0/null')
            rows=[]
            for f,m,t in zip(facts,metadata,terms):
                ident=f['fact_id'];v=f['raw_value'];group=m['group'];amount=ident in ['BUREAU_C1_CREDIT_SUM','BUREAU_C1_CREDIT_MEAN','BUREAU_C1_CREDIT_MAX']
                if group=='application':
                    fact_text=f'保存输入显示，{f["label"]}为{formatted(v)}。' if v is not None else f'保存输入中的{f["label"]}为空（null），未补写原值。'
                    missing_text='原值可用；0按有效来源编码保留。' if v is not None else '来源保存值为空，不能自动解释为0。'
                else:
                    n=values['BUREAU_ELIGIBLE_COUNT' if group=='bureau' else 'PREV_ELIGIBLE_COUNT'];name='征信借款' if group=='bureau' else '历史申请'
                    if n==0:
                        fact_text=f'保存汇总显示，当前规则下无合格{name}记录；{f["label"]}保留'+('0。' if v==0 else '为空（null）。')
                        missing_text='这是无合格历史：计数／存在标记为0，金额／比例／距今天数为null。'
                    elif v is None:
                        fact_text=f'保存汇总中有{n}条合格{name}记录；{f["label"]}为空（null）。'
                        if amount:fact_text+=f'满足原币种与金额条件的记录数为{values["BUREAU_C1_AMOUNT_AVAILABLE_COUNT"]}；不将缺失金额写成0。'
                        missing_text='有合格历史与此字段为空分别判断。'
                    elif ident=='BUREAU_ACTIVE_FRACTION':
                        fact_text=f'保存汇总中，{n}条合格征信记录里Active状态有{values["BUREAU_ACTIVE_COUNT"]}条，占{v*100:.4f}%（原比例{formatted(v)}）。'
                        missing_text='分母非0；比例0是有效结果，与无历史的null不同。'
                    elif m['unit']=='days':
                        fact_text=f'保存汇总显示，最近合格{name}的'+('借款申请' if group=='bureau' else '决策')+f'发生在当前申请前{formatted(v)}天。'
                        missing_text='相对天数有效，未换算日历日期。'
                    elif m['unit']=='binary':
                        fact_text=f'保存汇总中有{n}条合格{name}记录，因此有合格历史标记为{formatted(v)}。'
                        missing_text='存在标记由合格记录集合决定。'
                    else:
                        fact_text=f'保存汇总中有{n}条合格{name}记录；{f["label"]}为{formatted(v)}'+('条。' if m['unit']=='records' else '（匿名currency 1金额尺度）。')
                        missing_text='原值为有效0。' if v==0 else '保存汇总值可用，未重新填补或缩放。'
                contribution=t['contribution_log_odds'];direction='positive' if contribution>0 else 'negative' if contribution<0 else 'zero'
                action='推高' if contribution>0 else '拉低' if contribution<0 else '未改变'
                response=f'当前树模型对此项的训练路径贡献为{contribution:+.4f}对数几率，{action}内部得分；与其余39项及参考值共同形成原预测。'
                if contribution==0:response+='这不表示该字段对所有申请都无作用。'
                evidence={'kind':'saved_model_input_only','verified_source_records':False,'source_table':f['source']['table'],'record_key_kind':m['source_key'],'application_id':case['case_key']['application_id'],'rule_version':f['rule']['version'],'saved_fact_id':ident}
                evidence_text='此处可回到该申请的保存模型输入；主表原始行尚未独立接入，不能称已核验原记录。'
                if group!='application':
                    evidence.update(kind='history_unavailable',record_query={'table':group,'feature':ident,'selection':'inputs'})
                    evidence_text='历史索引未载入：仅说明保存汇总，尚未核对来源记录及排除账本。'
                    if history is not None:
                        h=history.query(case['case_key']['population'],case['case_key']['application_id'],group,feature=ident,selection='inputs',limit=1,expected=case['case_revision'])
                        if any(values[k]!=value for k,value in h['summary_values'].items()):raise ValueError('Index/model values differ')
                        a=h['audit'];evidence.update(kind='history_index',verified_source_records=True,index_manifest_sha256=h['index_manifest_sha256'],source=h['source'],audit=a,matching_record_count=h['total'])
                        evidence_text=f'已核对索引：来源{a["raw_count"]}条，时间合格{a["eligible_count"]}条，时间排除{a["excluded_count"]}条；其中{a["feature_inputs_count"]}条进入本因素算式。'
                        if a['history_state']=='only_time_excluded':evidence_text+='有来源记录，但全部被时间规则排除。'
                        elif a['history_state']=='no_source_history':evidence_text+='当前索引范围未找到此类来源记录。'
                rows.append({'fact_id':ident,'label':f['label'],'raw_value':v,'unit':f['unit'],'unit_zh':m['unit_zh'],'missing_state':f['missing_state'],'missing_text':missing_text,
                             'definition_zh':m['definition_zh'],'formula':f['rule']['formula'],'time_zh':m['time_zh'],'fact_text':fact_text,'model_response_text':response,'review_text':m['review_zh'],
                             'contribution_log_odds':contribution,'direction':direction,'evidence_text':evidence_text,'evidence':evidence,'limits':self.templates['limits']})
            result={'case_key':case['case_key'],'case_revision':case['case_revision'],'sample_use':case['sample_use'],'version':VERSION,'method':METHOD_ZH,'contribution_method':METHOD,
                    'template_sha256':self.sha256,'dictionary_sha256':self.dictionary_sha256,'history_manifest_sha256':history.seal if history else None,'rows':rows,'feature_count':40}
            result['narrative_revision']=canonical_hash(result)
            return result
        except (KeyError,ValueError,TypeError,OverflowError) as exc:
            if isinstance(exc,NarrativeError):raise
            raise NarrativeError('NARRATIVE_SOURCE_MISMATCH') from exc
