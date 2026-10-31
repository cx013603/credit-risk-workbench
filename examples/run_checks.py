"""Run only the curated, synthetic, non-fitting suites in this snapshot."""
from pathlib import Path
import os
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tests'), str(ROOT)]


def main():
    if os.name == 'nt' and len(str(ROOT)) > 80:
        print('Windows: please place this snapshot in a shorter directory (root path <=80 characters). Some original Parquet fixture paths exceed Windows limits in deeply nested directories.', file=sys.stderr)
        return 2
    for folder in ['data/interim', 'reports/workbench']:
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    modules = ['test_history', 'test_workbench_features_v2', 'test_tree_explanation', 'test_workbench_experiment_verifier']
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(m) for m in modules)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
