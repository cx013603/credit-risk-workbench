"""Independent Python arithmetic for raw-case/fixture verification; no SQL reuse."""
from collections import defaultdict
import math

def reference_features(ids,previous,bureau,cc=(),pos=(),bb=(),inst=()):
    from .features_v2 import feature_definitions
    out={i:{f['name']:0 if f['kind']=='count' else None for f in feature_definitions()} for i in ids}
    prev={(p['SK_ID_CURR'],p['SK_ID_PREV']):p for p in previous}
    bureaus={p['SK_ID_BUREAU']:p for p in bureau}
    def before(v):return v is not None and math.isfinite(v) and v<0
    def previous_ok(r):
        p=prev.get((r['SK_ID_CURR'],r['SK_ID_PREV']));return p is not None and before(p['DAYS_DECISION'])
    def frac(a,b):return a/b if b else None
    for prefix,rows in [('CC',cc),('POS',pos)]:
        grouped=defaultdict(list)
        for r in rows:
            if r['SK_ID_CURR'] in out and previous_ok(r) and before(r['MONTHS_BALANCE']):grouped[r['SK_ID_CURR']].append(r)
        for i,rs in grouped.items():
            known=[r['SK_DPD'] for r in rs if r['SK_DPD'] is not None and math.isfinite(r['SK_DPD']) and r['SK_DPD']>=0 and int(r['SK_DPD'])==r['SK_DPD']]
            values=[len({r['SK_ID_PREV'] for r in rs}),len(rs),-max(r['MONTHS_BALANCE'] for r in rs),max(known) if known else None,frac(sum(v>0 for v in known),len(known)),len(rs)-len(known),sum(r['MONTHS_BALANCE']>=-6 for r in rs),sum(r['MONTHS_BALANCE']>=-12 for r in rs)]
            names=['LOAN_COUNT','MONTH_COUNT','MONTHS_SINCE_RECENT','MAX_DPD','POSITIVE_DPD_FRACTION','DPD_UNKNOWN_COUNT','LAST6_MONTH_COUNT','LAST12_MONTH_COUNT'];out[i].update({prefix+'_'+n:v for n,v in zip(names,values)})
    grouped=defaultdict(list)
    for r in bb:
        p=bureaus.get(r['SK_ID_BUREAU'])
        if p and p['SK_ID_CURR'] in out and before(p['DAYS_CREDIT']) and before(p['DAYS_CREDIT_UPDATE']) and before(r['MONTHS_BALANCE']):grouped[p['SK_ID_CURR']].append(r)
    for i,rs in grouped.items():
        known=[r for r in rs if r['STATUS'] in ['0','1','2','3','4','5']]
        values=[len({r['SK_ID_BUREAU'] for r in rs}),len(rs),-max(r['MONTHS_BALANCE'] for r in rs),frac(sum(r['STATUS']!='0' for r in known),len(known)),sum(r['STATUS'] not in ['0','1','2','3','4','5','C'] for r in rs)/len(rs),sum(r['STATUS']=='C' for r in rs)/len(rs),sum(r['MONTHS_BALANCE']>=-12 for r in rs)]
        names=['LOAN_COUNT','MONTH_COUNT','MONTHS_SINCE_RECENT','DELINQUENT_STATUS_FRACTION','UNKNOWN_STATUS_FRACTION','CLOSED_STATUS_FRACTION','LAST12_MONTH_COUNT'];out[i].update({'BB_'+n:v for n,v in zip(names,values)})
    grouped=defaultdict(list)
    for r in inst:
        if r['SK_ID_CURR'] in out and previous_ok(r) and before(r['DAYS_INSTALMENT']) and before(r['DAYS_ENTRY_PAYMENT']):grouped[r['SK_ID_CURR']].append(r)
    for i,rs in grouped.items():
        loans=defaultdict(list)
        for r in rs:loans[r['SK_ID_PREV']].append(r)
        excluded=0;groups=[]
        for loan,events in loans.items():
            versions={r['NUM_INSTALMENT_VERSION'] for r in events}
            if len(versions)!=1 or any(v is None or not math.isfinite(v) or v<=0 or int(v)!=v for v in versions):excluded+=1;continue
            g=defaultdict(list)
            for r in events:g[(r['NUM_INSTALMENT_VERSION'],r['NUM_INSTALMENT_NUMBER'],r['DAYS_INSTALMENT'])].append(r)
            groups.extend(g.values())
        delays=[max(r['DAYS_ENTRY_PAYMENT'] for r in g)-g[0]['DAYS_INSTALMENT'] for g in groups]
        names=['LOAN_COUNT','PAYMENT_RECORD_COUNT','DAYS_SINCE_RECENT_PAYMENT','EXCLUDED_CALENDAR_LOAN_COUNT','SINGLE_CALENDAR_GROUP_COUNT','LATE_PAYMENT_GROUP_FRACTION','MAX_PAYMENT_DELAY_DAYS','MULTI_PAYMENT_GROUP_FRACTION']
        values=[len(loans),len(rs),-max(r['DAYS_ENTRY_PAYMENT'] for r in rs),excluded,len(groups),frac(sum(v>0 for v in delays),len(groups)),max(max(v,0) for v in delays) if delays else None,frac(sum(len(g)>1 for g in groups),len(groups))]
        out[i].update({'INST_'+n:v for n,v in zip(names,values)})
    return out
