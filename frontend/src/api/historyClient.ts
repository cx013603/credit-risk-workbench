import type {CaseData, Envelope} from './types';
import {ApiError} from './client.ts';

export type HistoryTable = 'previous' | 'bureau';
export type HistorySelection = 'inputs' | 'eligible' | 'excluded' | 'all';
export type HistoryInfo = {status: 'loading' | 'ready' | 'not_connected' | 'failed'; index_manifest_sha256?: string; release_manifest_sha256?: string; dictionary_sha256?: string};
export type HistoryRecord = {source_key: number; normalized_projection_sha256: string; source_values: Record<string, string | number | null>; eligible: boolean; exclusion_reason: string; reason_text: string; feature_input_used: boolean; amount_state: string};
export type HistoryData = {case_key: CaseData['case_key']; case_revision: string; sample_use: string; table: HistoryTable; feature: string | null; selection: HistorySelection; index_version: string; index_manifest_sha256: string; policy: string; source: {table: string; normalized_parquet_sha256: string; original_sql_sha256: string; record_key: string}; summary_values: Record<string, number | null>; audit: {raw_count: number; eligible_count: number; excluded_count: number; reason_counts: Record<string, number>; feature_inputs_count: number; history_state: string}; items: HistoryRecord[]; total: number; offset: number; limit: number; next_offset: number | null};
export type HistoryQuery = {table: HistoryTable; feature?: string; selection: HistorySelection; offset: number; limit: number};
const hash = (v: unknown) => typeof v === 'string' && /^[a-f0-9]{64}$/.test(v);
const count = (v: unknown): v is number => typeof v === 'number' && Number.isSafeInteger(v) && v >= 0;
const invalid = (): never => {throw new ApiError('HISTORY_RESPONSE_INVALID', '历史资料未通过身份、版本或汇总值核对，请重新查询。')};

export function assertHistory(value: unknown, c: CaseData, info: HistoryInfo, q: HistoryQuery): HistoryData {
 try {
  const h = value as HistoryData;
  if (!h || h.case_key.population !== c.case_key.population || h.case_key.application_id !== c.case_key.application_id || h.case_revision !== c.case_revision || h.sample_use !== c.sample_use || h.table !== q.table || h.feature !== (q.feature || null) || h.selection !== q.selection || h.index_version !== 'workbench_history_index_v1' || h.policy !== 'history_before_day_zero_currency_separate_v1' || !hash(h.index_manifest_sha256) || h.index_manifest_sha256 !== info.index_manifest_sha256 || info.release_manifest_sha256 !== c.versions.release_manifest_sha256 || info.dictionary_sha256 !== c.versions.display_dictionary_sha256) invalid();
  const expected = c.profile.facts.filter(f => f.fact_id.startsWith(q.table === 'bureau' ? 'BUREAU_' : 'PREV_'));
  if (Object.keys(h.summary_values).length !== expected.length || expected.some(f => h.summary_values[f.fact_id] !== f.raw_value)) invalid();
  const a = h.audit;
  if (![a.raw_count, a.eligible_count, a.excluded_count, a.feature_inputs_count, h.total, h.offset, h.limit].every(count) || a.raw_count !== a.eligible_count + a.excluded_count || a.feature_inputs_count > a.eligible_count || h.offset !== q.offset || h.limit !== q.limit || !Object.values(a.reason_counts).every(count) || Object.values(a.reason_counts).reduce((x, y) => x + y, 0) !== a.raw_count) invalid();
  if (h.total !== ({all: a.raw_count, eligible: a.eligible_count, excluded: a.excluded_count, inputs: a.feature_inputs_count}[q.selection]) || h.next_offset !== (q.offset + q.limit < h.total ? q.offset + q.limit : null)) invalid();
  if (!hash(h.source.normalized_parquet_sha256) || !hash(h.source.original_sql_sha256) || !Array.isArray(h.items) || h.items.length !== Math.max(0, Math.min(q.limit, h.total - q.offset)) || new Set(h.items.map(r => r.source_key)).size !== h.items.length) invalid();
  for (const r of h.items) {
   if (!count(r.source_key) || !hash(r.normalized_projection_sha256) || r.source_values.SK_ID_CURR !== c.case_key.application_id || r.source_values[q.table === 'bureau' ? 'SK_ID_BUREAU' : 'SK_ID_PREV'] !== r.source_key || typeof r.eligible !== 'boolean' || typeof r.feature_input_used !== 'boolean' || r.feature_input_used && !r.eligible || r.eligible !== (r.exclusion_reason === 'eligible') || typeof r.reason_text !== 'string' || Object.values(r.source_values).some(v => !(v === null || typeof v === 'string' || typeof v === 'number' && Number.isFinite(v)))) invalid();
   if (q.selection === 'inputs' && !r.feature_input_used || q.selection === 'eligible' && !r.eligible || q.selection === 'excluded' && r.eligible) invalid();
  }
  return h;
 } catch (e) {if (e instanceof ApiError) throw e; return invalid()}
}

export async function fetchHistory(c: CaseData, info: HistoryInfo, q: HistoryQuery, token: string, signal: AbortSignal): Promise<HistoryData> {
 const params = new URLSearchParams({selection: q.selection, offset: String(q.offset), limit: String(q.limit), expected_revision: c.case_revision});
 if (q.feature) params.set('feature', q.feature);
 let response: Response;
 try {response = await fetch(`/api/v2/cases/${c.case_key.population}/${c.case_key.application_id}/history/${q.table}?${params}`, {signal, headers: {'X-Selection-Token': token}})} catch (e) {if (signal.aborted) throw e; throw new ApiError('CONNECTION_FAILED', '无法连接历史资料服务，请重新查询。')}
 let body: Envelope;
 try {body = await response.json() as Envelope} catch {return invalid()}
 if (body?.schema_version !== 'workbench_api_v2.0' || body.selection_token !== token) invalid();
 if (!response.ok || body.error) throw new ApiError(body.error?.code || 'HISTORY_UNAVAILABLE', body.error?.message || '历史资料暂不可用。');
 if (body.status !== 'ready') invalid();
 return assertHistory(body.payload, c, info, q);
}
