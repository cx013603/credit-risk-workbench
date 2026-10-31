import {Button} from 'antd';
import type {Result} from './api/workspace71';
import {coverage,validationReading} from './reading71Data';
import {fieldLabel} from './presentation';
import {decimal,metric} from './format';
export function CoverageSummary({c,onDetail}:{c:Result;onDetail:()=>void}){
 const q=coverage(c.facts);
 return <section className="panel coverage-summary" data-testid="coverage-summary"><div className="panel-heading"><h2>资料缺口</h2><Button size="small" onClick={onDetail}>查看资料质量</Button></div><p><b>{q.observed}项已有观测</b> · {q.unavailable.length}项未提供或无法计算 · {q.noHistory.length}项无符合条件历史</p>{q.unavailable.length>0?<p>待核对：{q.unavailable.slice(0,3).map(f=>fieldLabel(f.name)).join('、')}{q.unavailable.length>3?'等。':'。'}</p>:<p>没有因来源缺失或无可用观测而留空的项目。</p>}<p className="fine">资料空白、有效0和无符合条件历史分别保留；无符合条件历史不能解读为没有借款。资料完整度不代表预测置信度。</p></section>
}
export function ModelReading({modelSha,threshold}:{modelSha:string;threshold:number}){
 const r=validationReading(modelSha,threshold);
 return <section className="model-reading" data-testid="model-reading"><p>预测的是公开数据集定义的还款困难事件，具体观察期限未披露。单笔概率是模型估计，不能直接当作已经证实的个人违约概率。</p>{r?<><p>分析阈值 {decimal(threshold*100)}% 来自保存的验证样本，用于研究分组，不是实际贷款的批准或拒绝标准。</p><details><summary>查看验证范围、阈值由来和误报漏报</summary><p>以下来自已有 {metric(r.rows)} 笔验证申请，含 {metric(r.events)} 笔困难事件标签。该批数据参与过模型比较和阈值选择，尚无新增独立最终测试。</p><div className="table-scroll"><table><thead><tr><th>来源记录</th><th>阈值以上</th><th>阈值以下</th></tr></thead><tbody><tr><th>有困难事件标签</th><td>识别 {metric(r.confusion.TP)} 笔</td><td>漏报 {metric(r.confusion.FN)} 笔</td></tr><tr><th>其他申请</th><td>误报 {metric(r.confusion.FP)} 笔</td><td>排除 {metric(r.confusion.TN)} 笔</td></tr></tbody></table></div><p>困难事件识别率 {decimal(r.recall*100)}%（以有困难标签的申请为分母）；其他申请误报率 {decimal(r.falsePositive*100)}%（以其他申请为分母）。阈值以上的申请中，困难标签占 {decimal(r.precision*100)}%，分母不同。</p><p>选择让“识别率减去误报率”最大的切分点，差为 {decimal(r.ks*100)} 个百分点（KS）。它没有考虑真实授信的收益和成本，也不会随新增待分析申请自动改变。</p><p>排序能力 AUC 为 {r.roc_auc.toFixed(4)}，不等于准确率 {(r.roc_auc*100).toFixed(2)}%。AUC衡量这批样本中的排序区分能力，不能证明单笔概率准确，也不能证明未来申请表现相同。</p></details></>:<p>当前模型没有匹配的保存验证说明，不展示其他版本的指标。</p>}</section>
}
export function UsageGuide(){return <section className="drawer-section" data-testid="usage-guide"><h3>第一次使用</h3><ol><li>在“申请分析”输入申请来源和编号，查看当前申请的概况及预测。</li><li>查看“主要影响因素”和“资料缺口”。需要核对来源时点击因素，需要完整资料时展开71项列表。</li><li>用“预览报告”阅读，用“导出报告（HTML）”保存；在浏览器打印窗口可选择另存为PDF。</li></ol><h3>怎样读这份分析</h3><p>有车和有房产只在来源明确时展示；居住类型不能证明产权。时间相对于本次申请，金额保留来源单位。</p><p>未提供不等于0；无符合条件历史不等于没有借款；资料完整不等于预测可靠。模型影响表示本次计算响应，不能当作原因或审批意见。</p><h3>保存与继续分析</h3><p>“复核与报告”保存自己的备注和固定快照；有未保存修改时先保存或放弃，再换编号。“新申请资料”用于填写、检查、保存完整资料后分析。</p></section>}
