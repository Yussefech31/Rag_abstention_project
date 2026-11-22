"""
tests/test_features.py
Unit tests for 10-dimensional feature engineering in RAG Abstention.
"""

import unittest
import numpy as np
import pandas as pd
from src.pipeline import (
    FEATURES,
    tokenize,
    token_set,
    Passage,
    RAGAbstentionPipeline,
    BM25Retriever,
)


class TestFeatureEngineering(unittest.TestCase):

    def test_feature_list_exact_count(self):
        """Verify that exactly 10 inference-available features are defined."""
        self.assertEqual(len(FEATURES), 10)

    def test_source_passage_retrieved_excluded(self):
        """
        Verify that source_passage_retrieved is strictly excluded from predictive features
        to prevent data leakage / runtime unavailability.
        """
        self.assertNotIn("source_passage_retrieved", FEATURES)

    def test_expected_feature_names(self):
        """Verify the exact required 10 feature names."""
        expected = [
            "top_bm25_score",
            "second_bm25_score",
            "bm25_score_gap",
            "mean_topk_bm25",
            "qa_score",
            "qa_no_answer_probability",
            "predicted_answer_length",
            "answer_nonempty",
            "question_length",
            "question_token_count",
        ]
        self.assertEqual(FEATURES, expected)

    def test_tokenization_helpers(self):
        """Test regex tokenization and token set functions."""
        text = "Who is the President of the United States in 2026?"
        tokens = tokenize(text)
        self.assertIn("who", tokens)
        self.assertIn("president", tokens)
        self.assertIn("2026", tokens)

        t_set = token_set(text)
        self.assertIsInstance(t_set, set)
        self.assertIn("president", t_set)

    def test_extract_features_structure_and_types(self):
        """Verify that extract_features produces a DataFrame with exact columns and numeric dtypes."""
        dummy_df = pd.DataFrame([{
            "paragraph_id": "p1",
            "article_title": "Test Article",
            "context": "The quick brown fox jumps over the lazy dog."
        }])
        retriever = BM25Retriever(dummy_df)
        pipe = RAGAbstentionPipeline(retriever=retriever)

        passages = [
            Passage(paragraph_id="p1", article_title="Test", context="sample context", bm25_score=15.5, rank=1),
            Passage(paragraph_id="p2", article_title="Test2", context="sample context 2", bm25_score=10.5, rank=2),
        ]
        qa_pred = {
            "answer": "quick brown fox",
            "qa_score": 0.88,
            "qa_no_answer_probability": 0.05,
        }

        feat_df, feat_dict = pipe.extract_features(
            question="What jumps over the lazy dog?",
            retrieved_passages=passages,
            qa_pred=qa_pred,
        )

        self.assertEqual(list(feat_df.columns), FEATURES)
        self.assertEqual(len(feat_df), 1)
        self.assertAlmostEqual(feat_dict["top_bm25_score"], 15.5)
        self.assertAlmostEqual(feat_dict["second_bm25_score"], 10.5)
        self.assertAlmostEqual(feat_dict["bm25_score_gap"], 5.0)
        self.assertAlmostEqual(feat_dict["qa_score"], 0.88)
        self.assertEqual(feat_dict["answer_nonempty"], 1.0)
        self.assertEqual(feat_dict["predicted_answer_length"], 3.0)


if __name__ == "__main__":
    unittest.main()
