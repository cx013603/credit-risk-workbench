import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {assertHistory, fetchHistory} from '../src/api/historyClient.ts';
const receipt=JSON.parse(readFileSync(new URL('../../reports/workbench/step5-20261028/http01.json',import.meta.url),'utf8'));
const c=receipt.cases[0],info=receipt.index,h=receipt.details[8];
const q={table:'previous',feature:'PREV_REVOLVING_COUNT',selection:'inputs',offset:0,limit:1};

test('real history response binds source identity, revision, model inputs and index',()=>{assert.equal(assertHistory(h,c,info,q).items[0].source_key,1489396)});
test('wrong case, source, revision or index cannot render',()=>{for(const change of [x=>x.case_key.application_id=100003,x=>x.case_key.population='unlabeled',x=>x.case_revision='0'.repeat(64),x=>x.index_manifest_sha256='0'.repeat(64)]){const x=structuredClone(h);change(x);assert.throws(()=>assertHistory(x,c,info,q))}});
test('changed summary, null instead of zero and broken audit are rejected',()=>{for(const change of [x=>x.summary_values.PREV_REVOLVING_COUNT=3,x=>x.summary_values.PREV_OTHER_CONTRACT_COUNT=null,x=>x.audit.raw_count=10,x=>x.next_offset=null,x=>x.total=3]){const x=structuredClone(h);change(x);assert.throws(()=>assertHistory(x,c,info,q))}});
test('foreign record, nonfinite value and mismatched filter are rejected',()=>{for(const change of [x=>x.items[0].source_values.SK_ID_CURR=100003,x=>x.items[0].source_values.DAYS_DECISION=NaN,x=>x.items[0].normalized_projection_sha256='bad',x=>x.items[0].feature_input_used=false]){const x=structuredClone(h);change(x);assert.throws(()=>assertHistory(x,c,info,q))}});
test('history selection-token mismatch is rejected',async()=>{const old=globalThis.fetch;globalThis.fetch=async()=>({ok:true,json:async()=>({schema_version:'workbench_api_v2.0',selection_token:'old',status:'ready',payload:h,error:null})});try{await assert.rejects(fetchHistory(c,info,q,'current',new AbortController().signal),/核对/)}finally{globalThis.fetch=old}});
test('history service failure returns explicit error without data',async()=>{const old=globalThis.fetch;globalThis.fetch=async()=>({ok:false,json:async()=>({schema_version:'workbench_api_v2.0',selection_token:'current',status:'failed',payload:null,error:{code:'HISTORY_INDEX_UNAVAILABLE',message:'历史索引暂不可用'}})});try{await assert.rejects(fetchHistory(c,info,q,'current',new AbortController().signal),/历史索引暂不可用/)}finally{globalThis.fetch=old}});
