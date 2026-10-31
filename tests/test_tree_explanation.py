"""Small independently calculated trees: counts, routes, categories and additivity."""
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock
import numpy as np
from sklearn.ensemble._hist_gradient_boosting.common import PREDICTOR_RECORD_DTYPE
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from credit_risk.raw import DataError
from credit_risk.tree_explanation import node_references, walk_tree
from credit_risk.tree_explanation_page import render_page


def stump():
    n = np.zeros(3, dtype=PREDICTOR_RECORD_DTYPE)
    n['count'] = [4, 1, 3]; n['is_leaf'] = [0, 1, 1]
    n['value'] = [99., -.6, .2]  # Internal stored update must not become a reference.
    n[0]['left'] = 1; n[0]['right'] = 2; n[0]['bin_threshold'] = 5
    n[0]['missing_go_to_left'] = 1
    return n


class PathTests(unittest.TestCase):
    def test_count_weighted_reference_uses_final_leaves(self):
        np.testing.assert_allclose(node_references(stump(), 1), [0., -.6, .2], atol=1e-15)

    def test_numeric_equality_and_missing_left(self):
        n = stump(); refs = node_references(n, 1)
        terms, leaves = walk_tree(n, np.zeros((0, 8), dtype=np.uint32), refs,
                                 np.array([[0], [5], [6], [255]], dtype=np.uint8), 255)
        np.testing.assert_allclose(leaves, [-.6, -.6, .2, -.6])
        np.testing.assert_allclose(refs[0]+terms.sum(axis=1), leaves)

    def test_missing_right(self):
        n = stump(); n[0]['missing_go_to_left'] = 0
        _, leaves = walk_tree(n, np.zeros((0, 8), dtype=np.uint32), node_references(n, 1),
                             np.array([[255]], dtype=np.uint8), 255)
        self.assertEqual(leaves[0], .2)

    def test_categories_across_bitset_words(self):
        n = stump(); n[0]['is_categorical'] = 1; n[0]['missing_go_to_left'] = 0
        bits = np.zeros((1, 8), dtype=np.uint32); bits[0, 0] = 1 << 2; bits[0, 1] = 1 << 2
        terms, leaves = walk_tree(n, bits, node_references(n, 1), np.array([[2], [3], [34], [255]], dtype=np.uint8), 255)
        np.testing.assert_allclose(leaves, [-.6, .2, -.6, .2]); np.testing.assert_allclose(terms[:, 0], leaves)

    def test_repeated_feature_telescopes_and_unused_stays_zero(self):
        n = np.zeros(5, dtype=PREDICTOR_RECORD_DTYPE)
        n['count'] = [8, 4, 4, 2, 2]; n['is_leaf'] = [0, 1, 0, 1, 1]
        n['value'] = [100., -2., 100., 0., 4.]
        n[0]['left'] = 1; n[0]['right'] = 2; n[0]['bin_threshold'] = 4
        n[2]['left'] = 3; n[2]['right'] = 4; n[2]['bin_threshold'] = 8
        refs = node_references(n, 2)
        terms, leaves = walk_tree(n, np.zeros((0, 8), dtype=np.uint32), refs, np.array([[0, 1], [6, 2], [9, 3]], dtype=np.uint8), 255)
        np.testing.assert_allclose(terms[:, 0], [-2., 0., 4.]); np.testing.assert_array_equal(terms[:, 1], 0.)
        np.testing.assert_allclose(refs[0]+terms.sum(axis=1), leaves)

    def test_constant_tree(self):
        n = np.zeros(1, dtype=PREDICTOR_RECORD_DTYPE); n[0]['count'] = 3; n[0]['is_leaf'] = 1; n[0]['value'] = .7
        refs = node_references(n, 2)
        terms, leaves = walk_tree(n, np.zeros((0, 8), dtype=np.uint32), refs, np.array([[0, 255]], dtype=np.uint8), 255)
        np.testing.assert_array_equal(terms, [[0., 0.]]); self.assertEqual(leaves[0], .7)

    def test_bad_counts(self):
        n = stump(); n[0]['count'] = 5
        with self.assertRaises(DataError): node_references(n, 1)

    def test_bad_feature_index(self):
        n = stump(); n[0]['feature_idx'] = 1
        with self.assertRaises(DataError): node_references(n, 1)

    def test_unreachable_node(self):
        n = np.concatenate([stump(), stump()[1:2]])
        with self.assertRaises(DataError): node_references(n, 1)

    def test_nonfinite_leaf(self):
        n = stump(); n[1]['value'] = np.nan
        with self.assertRaises(DataError): node_references(n, 1)

    def test_invalid_category_bitset(self):
        n = stump(); n[0]['is_categorical'] = 1
        with self.assertRaises(DataError):
            walk_tree(n, np.zeros((0, 8), dtype=np.uint32), node_references(n, 1), np.array([[2]], dtype=np.uint8), 255)

    def test_page_rejects_duplicate_and_extra_parameters(self):
        s = Mock()
        for query in ['population=labeled&id=1&id=2', 'population=labeled&id=1&x=2', 'id=1', 'population=labeled&id=abc']:
            with self.subTest(query=query):
                status, html = render_page(s, query); self.assertEqual(status, 400); self.assertNotIn('data-id=', html)
        s.explain_ids.assert_not_called()

    def test_unknown_id_clears_result_and_escapes_input(self):
        s = Mock(); s.explain_ids.side_effect = DataError('tree_explanation_unknown_id', 'unknown')
        status, html = render_page(s, 'population=labeled&id=999')
        self.assertEqual(status, 404); self.assertNotIn('data-id=', html)
        status, html = render_page(s, 'population=labeled&id=%3Cscript%3E')
        self.assertEqual(status, 400); self.assertNotIn('<script>', html); self.assertIn('&lt;script&gt;', html)


if __name__ == '__main__': unittest.main()
