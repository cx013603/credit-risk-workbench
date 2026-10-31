-- 仅关联两张已聚合、申请键唯一的摘要；不会将两张一对多历史明细直接连接。
-- {{history_columns}} 由已验证的字段契约生成；计数的0是结构定义，金额／比率／距今天数保持null。
SELECT a.* EXCLUDE (file_row_number), {{history_columns}}
FROM application_input a
LEFT JOIN bureau_features b USING (SK_ID_CURR)
LEFT JOIN previous_features p USING (SK_ID_CURR)
ORDER BY a.file_row_number;
