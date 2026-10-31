-- cc_input / pos_input contain only allowed application IDs and typed raw rows.
-- {prefix} / {input} are a closed internal substitution, never user identifiers.
CREATE TEMP TABLE {prefix}_eligible AS
SELECT h.* FROM {input} h JOIN previous_raw p
ON h.SK_ID_PREV=p.SK_ID_PREV AND h.SK_ID_CURR=p.SK_ID_CURR
WHERE p.DAYS_DECISION<0 AND h.MONTHS_BALANCE<0;

CREATE TEMP TABLE {prefix}_loan AS
SELECT SK_ID_CURR,SK_ID_PREV,COUNT(*) n,MAX(MONTHS_BALANCE) latest,
 MAX(SK_DPD) FILTER(WHERE SK_DPD>=0 AND SK_DPD=floor(SK_DPD)) max_dpd,
 COUNT(*) FILTER(WHERE SK_DPD>=0 AND SK_DPD=floor(SK_DPD)) known,
 COUNT(*) FILTER(WHERE SK_DPD>0 AND SK_DPD=floor(SK_DPD)) positive,
 COUNT(*) FILTER(WHERE MONTHS_BALANCE>=-6) last6,
 COUNT(*) FILTER(WHERE MONTHS_BALANCE>=-12) last12
FROM {prefix}_eligible GROUP BY SK_ID_CURR,SK_ID_PREV;

CREATE TABLE {prefix}_features AS
SELECT SK_ID_CURR,COUNT(*)::BIGINT {prefix}_LOAN_COUNT,
 SUM(n)::BIGINT {prefix}_MONTH_COUNT,-MAX(latest)::DOUBLE {prefix}_MONTHS_SINCE_RECENT,
 MAX(max_dpd)::DOUBLE {prefix}_MAX_DPD,
 SUM(positive)::DOUBLE/NULLIF(SUM(known),0) {prefix}_POSITIVE_DPD_FRACTION,
 SUM(n-known)::BIGINT {prefix}_DPD_UNKNOWN_COUNT,
 SUM(last6)::BIGINT {prefix}_LAST6_MONTH_COUNT,SUM(last12)::BIGINT {prefix}_LAST12_MONTH_COUNT
FROM {prefix}_loan GROUP BY SK_ID_CURR;
