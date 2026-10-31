-- default_flag: 1 表示在样例设定的后续 12 个月内违约，0 表示未违约。
-- 100.0 保证按浮点数计算百分比。
SELECT
    COUNT(*) AS application_count,
    SUM(default_flag) AS default_count,
    ROUND(100.0 * SUM(default_flag) / COUNT(*), 2) AS default_rate_pct
FROM applications;
