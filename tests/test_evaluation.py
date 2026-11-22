"""
tests/test_evaluation.py
Unit tests for SQuAD normalization, Exact Match, Token F1, and Answerability metrics.
"""

import unittest
import numpy as np
from scripts.run_experiments import (
    normalize_answer,
    exact_match,
    token_f1,
    best_reference_score,
)
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


class TestEvaluationMetrics(unittest.TestCase):

    def test_normalize_answer(self):
        """Test lowercase, punctuation removal, article removal, and whitespace normalization."""
        self.assertEqual(normalize_answer("The United States!"), "united states")
        self.assertEqual(normalize_answer("   An   apple   "), "apple")
        self.assertEqual(normalize_answer("A, B, and C."), "b and c")

    def test_exact_match(self):
        """Test exact match scoring."""
        self.assertEqual(exact_match("Tullamarine Airport", "tullamarine airport"), 1)
        self.assertEqual(exact_match("The Tullamarine Airport", "Tullamarine Airport"), 1)
        self.assertEqual(exact_match("Melbourne Airport", "Tullamarine Airport"), 0)

    def test_token_f1(self):
        """Test token-level overlap F1 score."""
        # Exact match
        self.assertAlmostEqual(token_f1("Barack Obama", "Barack Obama"), 1.0)
        # Disjoint
        self.assertAlmostEqual(token_f1("Paris", "London"), 0.0)
        # Partial overlap (2 tokens pred, 3 tokens gold, 2 common -> P = 2/2 = 1.0, R = 2/3, F1 = 2*1*(2/3)/(1 + 2/3) = 0.8)
        score = token_f1("Albert Einstein", "Professor Albert Einstein")
        self.assertAlmostEqual(score, 0.8)

    def test_best_reference_score(self):
        """Test selecting the maximum score across multiple valid reference answers."""
        refs = [{"text": "Tullamarine"}, {"text": "Tullamarine Airport"}]
        em = best_reference_score("Tullamarine Airport", refs, exact_match)
        self.assertEqual(em, 1.0)

    def test_answerability_metrics_calculation(self):
        """Test precision, recall, F1, accuracy, FAR, and FAbR calculation on known synthetic labels."""
        # 4 answerable, 4 unanswerable
        y_true = np.array([True, True, True, True, False, False, False, False])
        # Model accepts 3 of 4 answerable (1 false abstention), accepts 1 of 4 unanswerable (1 false answer)
        decisions = np.array([True, True, True, False, True, False, False, False])

        p, r, f1, _ = precision_recall_fscore_support(y_true, decisions, average="binary")
        acc = accuracy_score(y_true, decisions)

        # TP = 3, FP = 1, FN = 1, TN = 3
        # Precision = 3/4 = 0.75, Recall = 3/4 = 0.75, F1 = 0.75, Acc = 6/8 = 0.75
        self.assertAlmostEqual(p, 0.75)
        self.assertAlmostEqual(r, 0.75)
        self.assertAlmostEqual(f1, 0.75)
        self.assertAlmostEqual(acc, 0.75)

        # FAR = FP / (FP + TN) = 1/4 = 0.25
        unans = ~y_true
        far = float(decisions[unans].mean())
        self.assertAlmostEqual(far, 0.25)

        # FAbR = FN / (TP + FN) = 1/4 = 0.25
        ans = y_true
        fabr = float((~decisions[ans]).mean())
        self.assertAlmostEqual(fabr, 0.25)


if __name__ == "__main__":
    unittest.main()
