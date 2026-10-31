import sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from credit_risk.evaluation import validation_report
from scripts.verify_workbench_experiment_v2 import independent_metrics,check_metrics

class IndependentMetricsTests(unittest.TestCase):
    def test_tied_predictions_and_highest_KS_threshold(self):
        y=np.array([0,0,1,1]);p=np.array([.1,.4,.35,.8]);r=independent_metrics(y,p)
        self.assertEqual(r['roc_auc'],.75);self.assertEqual(r['ks'],.5);self.assertEqual(r['threshold'],.8)
    def test_probability_edges_and_exact_ten_bins(self):
        p=np.r_[0,np.linspace(0,1,11),1];y=np.array([0,1,0,1,0,1,0,1,0,1,0,1,0])
        check_metrics(y,p,validation_report(y,p),True)
    def test_many_ties_independent_ranks_match_reference(self):
        rng=np.random.default_rng(14);y=rng.integers(0,2,2000);p=rng.integers(0,11,2000)/10
        check_metrics(y,p,validation_report(y,p),True)
    def test_order_does_not_change_metric_values(self):
        y=np.array([1,0,1,0,0,1]);p=np.array([.3,.3,.9,.1,.7,.3]);r=independent_metrics(y,p)
        self.assertEqual(r,independent_metrics(y[::-1],p[::-1]))
if __name__=='__main__':unittest.main()
