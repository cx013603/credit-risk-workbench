import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {coverage,validationReading} from '../src/reading71Data.ts';
import {category,factValue,fieldLabel} from '../src/presentation.ts';
const raw=JSON.parse(fs.readFileSync(new URL('../../reports/workbench/deploy71-20261030/http04/http.json',import.meta.url)));
const c=raw.cases[0];
test('housing and closed credit do not invent ownership or repayment',()=>{
 assert.equal(category('House / apartment'),'住宅／公寓');assert.equal(category('Closed'),'已关闭');assert.doesNotMatch(fieldLabel('BUREAU_CLOSED_COUNT'),/结清/);
});
test('all elapsed times use this application as reference',()=>{
 assert.equal(factValue('INST_DAYS_SINCE_RECENT_PAYMENT',12),'申请前12天');assert.equal(factValue('CC_MONTHS_SINCE_RECENT',1),'申请前1个月');assert.doesNotMatch(fieldLabel('BUREAU_DAYS_SINCE_MOST_RECENT'),/距今/);
});
test('zero, missing observation and no eligible history remain separate',()=>{
 const q=coverage([{name:'zero',value:0,missing_state:'observed'},{name:'null',value:null,missing_state:'source_null_or_no_eligible_observation'},{name:'historyZero',value:0,missing_state:'no_eligible_history'},{name:'historyNull',value:null,missing_state:'no_eligible_history'}]);
 assert.equal(q.observed,1);assert.equal(q.unavailable.length,1);assert.equal(q.noHistory.length,2);assert.equal(factValue('BUREAU_ACTIVE_FRACTION',null,'source_null_or_no_eligible_observation'),'未提供或无可用记录');
});
test('validation figures are bound to this exact model and threshold',()=>{
 const r=validationReading(c.model_sha256,c.threshold);assert.equal(r.rows,46127);assert.equal(r.confusion.TP,2460);assert.equal(r.confusion.FP,15518);assert.equal(r.confusion.FN,1264);assert.equal(r.confusion.TN,26885);assert.equal(r.events,3724);assert.equal(r.independent_final_test,false);assert.equal(r.recall,2460/3724);assert.equal(r.precision,2460/17978);assert.equal(validationReading('different',c.threshold),null);assert.equal(validationReading(c.model_sha256,.1),null);
});
test('coverage partitions all 71 saved facts for every existing snapshot',()=>{
 for(const c of raw.cases){const q=coverage(c.facts);assert.equal(q.observed+q.unavailable.length+q.noHistory.length,71)}
});
