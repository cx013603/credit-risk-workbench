"""统一二元风险指标；调用方控制样本用途，本模块不读取任何分区。"""
import numpy as np
from sklearn.metrics import brier_score_loss, confusion_matrix, roc_auc_score, roc_curve
from .raw import DataError


def checked_probabilities(y, probability):
    y, p = np.asarray(y), np.asarray(probability, dtype=float)
    if (y.ndim != 1 or p.shape != y.shape or not len(y) or set(np.unique(y)) != {0, 1}
            or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any()):
        raise DataError('evaluation_invalid', 'Both binary labels and finite probabilities in [0,1] required')
    return y.astype(int), p


def ranking_metrics(y, probability):
    y, p = checked_probabilities(y, probability)
    fpr, tpr, thresholds = roc_curve(y, p, drop_intermediate=False)
    finite = np.isfinite(thresholds)
    return {'rows': len(y), 'events': int(y.sum()), 'event_rate': float(y.mean()),
            'roc_auc': float(roc_auc_score(y, p)), 'ks': float(np.max((tpr-fpr)[finite])),
            'brier_score': float(brier_score_loss(y, p))}


def validation_report(y, probability):
    """按冻结政策在验证集选最大KS阈值；均匀10箱仅诊断概率，不拟合校准器。"""
    y, p = checked_probabilities(y, probability)
    result = ranking_metrics(y, p)
    fpr, tpr, thresholds = roc_curve(y, p, drop_intermediate=False)
    finite = np.isfinite(thresholds)
    values, ts = (tpr-fpr)[finite], thresholds[finite]
    threshold = float(np.max(ts[values == values.max()]))
    predicted = p >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0,1]).ravel().tolist()
    selected = tp+fp
    result['threshold'] = {'scope':'validation_only', 'rule':'max_ks_tie_highest_finite_threshold',
        'value':threshold, 'TP':tp,'FP':fp,'TN':tn,'FN':fn,
        'precision':tp/selected if selected else 0., 'recall':tp/(tp+fn),
        'false_positive_rate':fp/(fp+tn), 'selected_fraction':selected/len(y),
        'meaning':'classroom risk flag only; not lending approval or economic optimality'}
    bins = np.searchsorted(np.linspace(0,1,11)[1:-1], p)
    result['calibration_bins'] = [{'bin':i, 'rows':int((bins==i).sum()),
        'mean_prediction':float(p[bins==i].mean()) if (bins==i).any() else None,
        'event_fraction':float(y[bins==i].mean()) if (bins==i).any() else None} for i in range(10)]
    return result


def prediction_agreement(expected, actual):
    a, b = np.asarray(expected), np.asarray(actual)
    if (a.shape != b.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all()
            or ((a < 0) | (a > 1) | (b < 0) | (b > 1)).any()):
        raise DataError('prediction_invalid', 'Incompatible or invalid probability arrays')
    difference = float(np.max(np.abs(a-b)))
    if difference > 1e-10:
        raise DataError('prediction_mismatch', 'Probability difference exceeds 1e-10')
    return {'passed':True,'max_absolute_difference':difference,'absolute_tolerance':1e-10}
