"""Sealed train-only descriptive reference; standard-library runtime only."""
from bisect import bisect_right
import hashlib
import json
import math
from pathlib import Path
from .contracts import METHOD

VERSION = 'workbench_train_reference_v1'
RARE_SHARE = .01
class ReferenceError(ValueError): pass
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def check(value):
    if not value: raise ReferenceError('TRAIN_REFERENCE_INVALID')
def finite(v): return type(v) in (int, float) and math.isfinite(v)
def canonical(v): return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def bucket(spec, value):
    if value is None: return 0
    check(isinstance(value,str) if spec['value_type']=='categorical' else finite(value))
    if spec['kind']=='all_missing': return 1
    if spec['kind']=='categorical':
        return spec['categories'].index(value)+1 if value in spec['categories'] else spec['buckets']-1
    if spec['kind']=='constant': return 1 if value<spec['constant'] else 2 if value==spec['constant'] else 3
    return bisect_right(spec['cuts'],value)+1
def fmt(v): return f'{v:.8g}'
def labels(s):
    if s['kind']=='all_missing': return ['缺失 null','训练中未见的非空值']
    if s['kind']=='categorical': return ['缺失 null',*s['categories'],'训练词表外类别']
    if s['kind']=='constant':
        v=fmt(s['constant']);return ['缺失 null',f'< {v}',f'= {v}',f'> {v}']
    cuts=s['cuts'];return ['缺失 null',f'< {fmt(cuts[0])}',*[f'[{fmt(a)}, {fmt(b)})' for a,b in zip(cuts,cuts[1:])],f'≥ {fmt(cuts[-1])}']

class TrainReference:
    def __init__(self,path,seal,release,dictionary,dictionary_path):
        try:
            check(digest(path)==seal)
            self.data=json.loads(Path(path).read_text(encoding='utf-8'));self.seal=seal
            self.dictionary=json.loads(Path(dictionary_path).read_text(encoding='utf-8'))
            d=self.data
            check(d['version']==VERSION and d['purpose']=='train_only_descriptive_reference' and d['population']=='labeled_train_applications')
            check(d['release_manifest_sha256']==release and d['dictionary_sha256']==dictionary and digest(dictionary_path)==dictionary)
            check(d['feature_columns']==self.dictionary['order'] and len(d['features'])==40 and list(d['features'])==d['feature_columns'])
            check(type(d['rows']) is int and d['rows']>0 and d['rare_share_threshold']==RARE_SHARE)
            check(d['validation']['passed'] and d['validation']['count_mismatches']==0 and d['validation']['training_ids_match'])
            for field in self.dictionary['fields']:
                s=d['features'][field['feature']];c=s['reference_counts']
                check(s['value_type']==('categorical' if field['type']=='string' else 'numeric') and s['kind'] in ['numeric','constant','all_missing','categorical'])
                check(type(s['buckets']) is int and 2<=s['buckets']<=512 and len(c)==s['buckets'] and all(type(x) is int and x>=0 for x in c) and sum(c)==d['rows'])
                if s['kind']=='categorical':check(s['value_type']=='categorical' and s['categories']==sorted(set(s['categories'])) and all(isinstance(x,str) for x in s['categories']) and s['buckets']==len(s['categories'])+2 and c[-1]==0)
                elif s['kind']=='numeric':check(s['value_type']=='numeric' and 1<=len(s['cuts'])<=9 and all(finite(x) for x in s['cuts']) and s['cuts']==sorted(set(s['cuts'])) and len(s['cuts'])+2==s['buckets'])
                elif s['kind']=='constant':check(s['value_type']=='numeric' and finite(s['constant']) and s['buckets']==4 and c[1]==c[3]==0)
                else:check(s['buckets']==2 and c[0]==d['rows'] and c[1]==0)
                if s['value_type']=='numeric' and s['kind']!='all_missing':check(finite(s['minimum']) and finite(s['maximum']) and s['minimum']<=s['maximum'])
                check(len(labels(s))==len(c))
        except (OSError,ValueError,KeyError,TypeError,IndexError) as exc:
            raise ReferenceError('TRAIN_REFERENCE_INVALID') from exc
    def info(self):
        d=self.data
        return {'status':'ready','version':VERSION,'reference_sha256':self.seal,'dictionary_sha256':d['dictionary_sha256'],'release_manifest_sha256':d['release_manifest_sha256'],'rows':d['rows'],'feature_count':40,'population':d['population'],'numeric_rule':d['numeric_rule'],'rare_share_threshold':RARE_SHARE,'percentile_status':'not_provided_binned_reference_only'}
    def render(self,c):
        d=self.data
        check(c['sample_use'] in ['train','validation','unlabeled_scoring'] and c['explanation']['method']==METHOD and c['explanation']['unit']=='log_odds')
        check(c['case_key']['population']==('unlabeled' if c['sample_use']=='unlabeled_scoring' else 'labeled'))
        for key in ['release_manifest_sha256','display_dictionary_sha256','feature_contract_sha256','split_manifest_sha256','tree_version','tree_run_id']:
            target=d['dictionary_sha256'] if key=='display_dictionary_sha256' else d[key] if key=='release_manifest_sha256' else d['output_versions'][key]
            check(c['versions'][key]==target)
        check([f['fact_id'] for f in c['profile']['facts']]==d['feature_columns'] and [t['fact_id'] for t in c['explanation']['terms']]==d['feature_columns'])
        for f,m in zip(c['profile']['facts'],self.dictionary['fields']):check(f['unit']==m['unit'] and f['label']==m['label'])
        e=c['explanation'];check(all(finite(t['contribution_log_odds']) for t in e['terms']))
        total=math.fsum(t['contribution_log_odds'] for t in e['terms']);score=e['base_log_odds']+total
        check(finite(score) and abs(score-e['raw_score'])<=1e-10 and abs(total-e['contribution_sum'])<=1e-10)
        probability=1/(1+math.exp(-score)) if score>=0 else math.exp(score)/(1+math.exp(score))
        check(abs(probability-c['model_outputs']['tree']['probability'])<=1e-10)
        rows=[]
        for f in c['profile']['facts']:
            s=d['features'][f['fact_id']];v=f['raw_value'];i=bucket(s,v);count=s['reference_counts'][i]
            kind='missing' if v is None else 'unseen_nonnull' if s['kind']=='all_missing' else 'unknown_category' if s['kind']=='categorical' and i==s['buckets']-1 else 'rare_category' if s['kind']=='categorical' and count/d['rows']<RARE_SHARE else 'outside_train_range' if s['value_type']=='numeric' and (v<s['minimum'] or v>s['maximum']) else 'observed'
            title={'missing':'当前值缺失，定位于独立缺失桶。','unknown_category':'当前类别未在训练参考词表中出现。','rare_category':'当前类别在训练参考中占比低于1%，单独保留。','unseen_nonnull':'训练参考全部为空，当前非空值没有可比较分布。','outside_train_range':'当前值超出训练参考最小值至最大值范围。','observed':'当前值已定位到训练参考桶。'}[kind]
            rows.append({'fact_id':f['fact_id'],'raw_value':v,'unit':f['unit'],'missing_state':f['missing_state'],'kind':s['kind'],'value_type':s['value_type'],'rows':d['rows'],'missing_count':s['reference_counts'][0],'bins':[{'index':j,'label':label,'count':n,'share':n/d['rows'],'current':j==i} for j,(label,n) in enumerate(zip(labels(s),s['reference_counts']))],'current_bucket':i,'position_status':kind,'current_bucket_count':count,'current_bucket_share':count/d['rows'],'minimum':s.get('minimum'),'maximum':s.get('maximum'),'percentile':None,'description':title+' 仅作训练申请分布对照，不是风险百分位或效果证明。'})
        top=sorted(e['terms'],key=lambda t:abs(t['contribution_log_odds']),reverse=True)[:5];ids={t['fact_id'] for t in top}
        rest=math.fsum(t['contribution_log_odds'] for t in e['terms'] if t['fact_id'] not in ids)
        result={'case_key':c['case_key'],'case_revision':c['case_revision'],'sample_use':c['sample_use'],'reference':self.info(),'rows':rows,'reconciliation':{'method':METHOD,'unit':'log_odds','base':e['base_log_odds'],'top':top,'rest_count':35,'rest':rest,'total':total,'raw_score':score,'probability':probability}}
        result['reference_revision']=canonical(result);return result
