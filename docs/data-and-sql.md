# 从历史明细到一笔申请一行

## 一个可以手算的例子

虚构申请 1 有 2 条符合时间条件的征信借款、2 条历史申请。两张明细直接连回主表会得到 2×2=4 行：征信金额 100 与 200 各出现两次，求和会从 300 变为 600。

本项目先按申请编号分别汇总两张历史表，再把摘要左关联到申请表。因此输入有多少笔申请，结果仍有多少笔；无匹配历史的申请也保留。

```sql
-- 示意：实际时间、币种、空值处理见链接中的完整 SQL。
WITH bureau_summary AS (
  SELECT SK_ID_CURR, COUNT(*) AS n, SUM(AMT_CREDIT_SUM) AS amount
  FROM bureau_input
  WHERE DAYS_CREDIT < 0 AND DAYS_CREDIT_UPDATE < 0
  GROUP BY SK_ID_CURR
)
SELECT a.SK_ID_CURR, b.n, b.amount
FROM application_input a
LEFT JOIN bureau_summary b USING (SK_ID_CURR);
```

[征信 SQL](../sql/02_bureau_features.sql)、[历史申请 SQL](../sql/03_previous_features.sql)与[最终关联](../sql/04_join_history.sql)保留实际实现。上面的示意省略了币种隔离等条件，不能代替完整实现。

## 时间、缺失和粒度检查

| 风险 | 实际处理 | 面试可以追问什么 |
|---|---|---|
| 同日和未来信息混入 | 征信时间与更新时间均严格小于 0；历史决策日严格小于 0 | 只有日粒度，为什么不把 0 当历史？ |
| 多种匿名币种金额相加 | currency 1 单独统计，其余／缺币种单列计数 | 为什么不换算成人民币？ |
| 缺失金额被当作 0 | 金额无可用观测时保持 null；有效金额 0 保留 | “没有借款”和“金额没记录”是否相同？ |
| 子表连接造成膨胀 | 独立聚合、摘要键唯一、主表 ID 集合和行数守恒 | 两类历史分别 3 条与 4 条时会怎样？ |
| 月度记录外键错误 | 同时匹配申请和父借款；检查重复月键和冲突支付记录 | 只按借款号关联够不够？ |
| 比例口径混淆 | 对符合条件的月记录计算比例，明确分母 | 先算各贷款比例再平均会有何不同？ |

原 40 项由 16 项主表字段和 24 项历史摘要组成。新 71 项保留原输入，再增加信用卡、POS、征信月度与分期还款相关的 31 项摘要。[新增特征实现](../src/credit_risk_workbench/features_v2.py)与[独立 Python 参考](../src/credit_risk_workbench/feature_reference_v2.py)可相互核对。

## 不依赖正式数据的验证

```bash
python examples/sql_walkthrough.py
```

示例包含 4 个虚构编号，输出 24 项历史特征。申请 1 展示连接膨胀与未来记录排除；申请 2 展示有历史但金额未知；申请 3 展示无历史；申请 4 展示有效 0。ID 为示例编号，不映射真实记录。

[原历史测试](../tests/test_history.py)覆盖手算、未来记录、无历史、原子写入和篡改拒绝；[新增历史测试](../tests/test_workbench_features_v2.py)覆盖 31 项 SQL 与独立参考、重复键、分期多次支付和训练折读取边界。它们验证具体实现行为，不证明新数据上的模型表现。
