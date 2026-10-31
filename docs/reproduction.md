# 怎样验证这份作品

## 环境

原项目实测 Windows x64、Python 3.14.5；数据依赖 DuckDB 2.0.0、PyArrow 26.0.0，模型依赖见根目录固定版本文件。版本来自原环境，不代表本次在所有平台重新安装验证。

建议在独立目录和虚拟环境中运行。本仓库的合成示例无需 Kaggle 登录、原始数据或保存模型，也不会启动后台或训练模型。

Windows 请使用短路径，例如 `C:/work/credit-risk`。原测试会生成带版本号的临时路径，深层目录可能触发系统长度限制；精选检查入口会在根路径超过80字符时给出提示。本轮已实际遇到此限制，随后在短目录复验，未修改原数据处理实现。

```bash
python -m venv .venv
```

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-data.txt
python examples/sql_walkthrough.py
python examples/metrics_walkthrough.py
```

macOS／Linux 使用 `source .venv/bin/activate`，其余命令相同；跨平台执行尚未在本轮实测。

## 可核对的预期结果

SQL 示例中，虚构申请1的两类合格历史分别2条：明细直接关联4行，征信金额错误求和600；正确聚合后仍是1行、征信合计300。4笔申请输出仍为4行。申请2金额null、申请3无历史、申请4金额0。脚本包含这些断言，任一不符会失败。

指标示例从保存聚合计数复算66.06%／36.60%／13.68%及38.98%；它不读取逐笔标签或推算AUC。

## 运行精选合成套件

```bash
python -m pip install -r requirements-model-smoke.txt
python examples/run_checks.py
```

该命令显式加载四份原项目测试，不运行全项目发现、不读取正式数据、不调用训练、不访问外部服务。临时文件在仓库内忽略的 `data/interim`、`reports` 下生成；测试结束清理其临时子目录。选定测试包括构造虚拟冻结包，用来验证“禁止读取原test”等行为，不能将其理解为真实冻结包已分发。

`config/experiment_plan.json` 在本包中是合成检查使用的哨兵文件，仅检查运行前后未被改动。它没有复制内部旧配置中的本机路径，也不是可用于正式实验的冻结协议。

可在 [review-checks.json](../evidence/review-checks.json) 看本次实际执行的环境、条数和结果。本机复用原依赖环境完成，未声称新机安装或托管 CI 已通过。

## 完整工作台与本仓库的关系

当前完整本机平台需要可信保存模型、原始／规范数据、训练参考、来源索引和版本绑定清单。它们包含大文件与使用数据，不属于公开作品包。源码中的运行编号和冻结哈希保留用于追溯，不要删除保护或编造清单来“跑通”。

因此本包交付的是代码审阅、合成验证与研究说明。完整本机端到端和同机恢复属于原工程已有核验范围；公开包不宣称一键重训、异机全量部署或线上可用。未提供公网地址，本机127.0.0.1地址也不是面试官能远程访问的演示站点。
