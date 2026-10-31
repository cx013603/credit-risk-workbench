-- 决策日严格早于当前申请。未来到期／终止字段、最终状态和无币种金额暂不作特征。
CREATE TEMP VIEW previous_classified AS
SELECT *, CASE
    WHEN DAYS_DECISION IS NULL THEN 'missing_decision_time'
    WHEN DAYS_DECISION >= 0 THEN 'decision_not_before'
    ELSE 'eligible'
END AS exclusion_reason
FROM previous_input;

CREATE TEMP VIEW previous_eligible AS
SELECT * EXCLUDE (exclusion_reason) FROM previous_classified WHERE exclusion_reason = 'eligible';

CREATE TABLE previous_features AS
SELECT SK_ID_CURR,
    COUNT(*)::BIGINT AS PREV_ELIGIBLE_COUNT,
    1::BIGINT AS PREV_HAS_ELIGIBLE_HISTORY,
    -MAX(DAYS_DECISION) AS PREV_DAYS_SINCE_MOST_RECENT,
    COUNT(*) FILTER (WHERE NAME_CONTRACT_TYPE = 'Cash loans') AS PREV_CASH_COUNT,
    COUNT(*) FILTER (WHERE NAME_CONTRACT_TYPE = 'Consumer loans') AS PREV_CONSUMER_COUNT,
    COUNT(*) FILTER (WHERE NAME_CONTRACT_TYPE = 'Revolving loans') AS PREV_REVOLVING_COUNT,
    COUNT(*) FILTER (WHERE NAME_CONTRACT_TYPE IS NULL OR NAME_CONTRACT_TYPE NOT IN ('Cash loans','Consumer loans','Revolving loans')) AS PREV_OTHER_CONTRACT_COUNT
FROM previous_eligible
GROUP BY SK_ID_CURR;
