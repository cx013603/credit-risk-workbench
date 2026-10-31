# 按问题阅读代码

这里是实际项目源码快照，原文件与导出文件的哈希对应关系见 [source-manifest.json](../evidence/source-manifest.json)。两处历史页面的页脚在导出时统一为来源说明，清单明确记录此呈现调整；原项目文件保持。保留代码中的版本、运行绑定和历史模块，是为了阅读与核查；被引用的数据、模型和原完整介质没有公开分发。

| 业务／技术问题 | 入口 | 重点看什么 |
|---|---|---|
| 原始表怎样读、错误怎样处理 | [raw.py](../src/credit_risk/raw.py)、[normalize.py](../src/credit_risk/normalize.py) | 主键、类型、数据范围与确定性处理 |
| 为什么先聚合再关联 | [history.py](../src/credit_risk/history.py)、[征信 SQL](../sql/02_bureau_features.sql)、[关联 SQL](../sql/04_join_history.sql) | 24 项契约、时间分类、结构性 0 与 null |
| 新增31项如何形成71项 | [features_v2.py](../src/credit_risk_workbench/features_v2.py)、[feature_reference_v2.py](../src/credit_risk_workbench/feature_reference_v2.py) | 父键、月度口径、分期支付分组及 SQL／Python 独立核对 |
| 训练处理怎样避免泄漏 | [logistic.py](../src/credit_risk/logistic.py)、[scorecard.py](../src/credit_risk/scorecard.py)、[tree.py](../src/credit_risk/tree.py) | 训练内词表／分箱，未知类别与全空列 |
| 对照实验如何控制预算和数据用途 | [experiment_v2.py](../src/credit_risk_workbench/experiment_v2.py) | 预定候选、32槽、train／validation标签访问边界 |
| AUC／KS／Brier怎样交叉核查 | [evaluation.py](../src/credit_risk/evaluation.py)、[独立指标核查](../scripts/verify_workbench_experiment_v2.py) | 并列值、最高KS阈值和概率边界 |
| 同一评分流程怎样用于不同入口 | [scoring.py](../src/credit_risk/scoring.py)、[candidate71.py](../src/credit_risk_workbench/candidate71.py) | 输入契约、特征顺序和保存模型绑定 |
| 树解释为什么能加总 | [tree_explanation.py](../src/credit_risk/tree_explanation.py) | 叶节点参考、缺失路由、重复特征的望远镜加总 |
| 新输入如何拒绝资料不完整 | [candidate71_contract.py](../src/credit_risk_workbench/candidate71_contract.py) | 未提供／无历史、父子键、有限数值与重复记录 |
| 当前产品怎样统一到71项 | [single71.tsx](../frontend/src/single71.tsx)、[workspace71.py](../src/credit_risk_workbench/workspace71.py) | 当前分析、筛选、比较、复核和报告的一致性 |
| 报告如何防止不可信文字执行 | [report71.py](../src/credit_risk_workbench/report71.py) | 纯文本转义、版本与贡献守恒检查 |
| 后台为何不重复加载模型 | [API](../src/credit_risk_workbench/api.py)、[worker](../src/credit_risk_workbench/worker.py) | 请求身份、超时、繁忙与失败返回 |
| 历史 API 和监控研究做了什么 | [scoring_api.py](../src/credit_risk/scoring_api.py)、[monitoring.py](../src/credit_risk/monitoring.py) | 有界批量、错误、缺失率和分布变化；属于原40项工程扩展 |
| 如何保护启动和恢复 | [launcher.py](../src/credit_risk_workbench/launcher.py)、[delivery.py](../src/credit_risk_workbench/delivery.py) | 进程身份、目录与哈希检查；依赖未分发的完整介质 |

## 本仓库可直接运行的合成检查

- [test_history.py](../tests/test_history.py)：手算、时间、币种、关联粒度、失败不覆盖和篡改拒绝。
- [test_workbench_features_v2.py](../tests/test_workbench_features_v2.py)：新增31项及固定折读取范围。
- [test_tree_explanation.py](../tests/test_tree_explanation.py)：手工树的路由与贡献守恒，不进行训练。
- [test_workbench_experiment_verifier.py](../tests/test_workbench_experiment_verifier.py)：合成标签／概率上的独立指标复算。

只运行明确列出的合成套件。全项目测试中存在需要正式产物的检查，本包不提供那些资产，也不承诺全量发现测试通过。前端源码和锁文件可以审阅；本轮未在另一台机器上重建整个平台。
