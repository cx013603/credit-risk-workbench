import test from 'node:test';
import assert from 'node:assert/strict';
import {factValue} from '../src/presentation.ts';
test('vehicle and property are explicit without invented quantities',()=>{
 assert.equal(factValue('FLAG_OWN_CAR','N'),'无车');
 assert.equal(factValue('FLAG_OWN_CAR','Y'),'有车（数量未提供）');
 assert.equal(factValue('FLAG_OWN_REALTY','Y'),'有房产（数量未提供）');
 assert.equal(factValue('FLAG_OWN_CAR',null),'车辆信息未提供');
 assert.equal(factValue('FLAG_OWN_CAR',2),'车辆情况待核对');
});
test('history counts and elapsed times carry their meanings on the list',()=>{
 assert.equal(factValue('PREV_ELIGIBLE_COUNT',9),'9条');
 assert.equal(factValue('INST_DAYS_SINCE_RECENT_PAYMENT',12),'申请前12天');
 assert.equal(factValue('HOUR_APPR_PROCESS_START',17),'17时');
 assert.equal(factValue('CC_MONTHS_SINCE_RECENT',1),'申请前1个月');
 assert.equal(factValue('CC_MONTH_COUNT',6),'6条借款月记录');
});
test('missing historical ratios do not become zero or no-debt conclusions',()=>{
 assert.equal(factValue('BUREAU_ACTIVE_FRACTION',null,'no_eligible_history'),'无符合条件的记录，无法计算');
 assert.equal(factValue('BUREAU_ACTIVE_FRACTION',0),'0.00%');
 assert.equal(factValue('BUREAU_ACTIVE_FRACTION',.25),'25.00%');
 assert.equal(factValue('BUREAU_ELIGIBLE_COUNT',0,'no_eligible_history'),'0条（无符合条件记录）');
});
test('income retains unknown currency and period, categories remain Chinese',()=>{
 assert.match(factValue('AMT_INCOME_TOTAL',135000),/135,000.*币种及收入周期未提供/);
 assert.equal(factValue('OCCUPATION_TYPE','Laborers'),'体力劳动人员');
 assert.equal(factValue('FLAG_EMP_PHONE',0),'无相关记录');
});
