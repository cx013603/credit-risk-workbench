"""Reading-only summary of the frozen model's existing validation evidence."""
import json
from pathlib import Path
EVIDENCE=json.loads(Path(__file__).with_name('model_reading71.json').read_text(encoding='utf-8'))
def render(model_sha,threshold):
    if model_sha!=EVIDENCE['model_sha256'] or threshold!=EVIDENCE['threshold']:return '<p>当前模型没有匹配的保存验证说明，不展示其他版本的指标。</p>'
    r=EVIDENCE;c=r['confusion'];tp,fp,tn,fn=[c[k] for k in ['TP','FP','TN','FN']]
    return f'''<h2>如何理解结果</h2><p>预测的是公开数据集定义的还款困难事件，具体观察期限未披露。概率是模型估计，不能直接当作已经证实的个人违约概率。</p><p>以下来自已有{r['rows']:,}笔验证申请，含{r['events']:,}笔困难标签；该批数据参与过模型比较和阈值选择，尚无新增独立最终测试。</p><table><thead><tr><th>来源记录</th><th>阈值以上</th><th>阈值以下</th></tr></thead><tbody><tr><th>有困难事件标签</th><td>识别{tp:,}笔</td><td>漏报{fn:,}笔</td></tr><tr><th>其他申请</th><td>误报{fp:,}笔</td><td>排除{tn:,}笔</td></tr></tbody></table><p>困难事件识别率{tp/(tp+fn)*100:.2f}%；其他申请误报率{fp/(fp+tn)*100:.2f}%。阈值以上申请中，困难标签占{tp/(tp+fp)*100:.2f}%，分母不同。</p><p>阈值{threshold*100:.2f}%选择让“识别率减去误报率”最大的切分点（KS差为{r['ks']*100:.2f}个百分点），未按真实授信收益和成本优化，不随新增待评分申请自动改变。AUC {r['roc_auc']:.4f}衡量排序区分能力，不等于准确率{r['roc_auc']*100:.2f}%，也不能证明单笔概率准确或未来效果。</p>'''
