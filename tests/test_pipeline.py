"""
tests/test_pipeline.py
Unit tests for RAG pipeline decisions, empty retrieval handling, and feature consistency.
"""

import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from src.pipeline import (
    RAGAbstentionPipeline,
    RAGResult,
    Passage,
    BM25Retriever,
    FEATURES,
    create_pipeline,
)


class MockRetriever:
    """Mock retriever returning predetermined passages."""

    def __init__(self, passages=None):
        self.passages = passages if passages is not None else []

    def retrieve(self, question, top_k=5):
        return self.passages[:top_k]


class MockReader:
    """Mock reader returning predetermined QA outputs."""

    def __init__(self, answer="Paris", qa_score=0.92, no_answer_prob=0.04):
        self.answer = answer
        self.qa_score = qa_score
        self.no_answer_prob = no_answer_prob

    def read_passages(self, question, passages):
        if not passages:
            return {"answer": "", "qa_score": 0.0, "qa_no_answer_probability": 1.0}, None
        best_p = passages[0]
        return {
            "answer": self.answer,
            "qa_score": self.qa_score,
            "qa_no_answer_probability": self.no_answer_prob,
        }, best_p


class TestRAGPipeline(unittest.TestCase):

    def setUp(self):
        self.sample_passages = [
            Passage(paragraph_id="p1", article_title="France", context="Paris is the capital of France.", bm25_score=22.5, rank=1),
            Passage(paragraph_id="p2", article_title="Europe", context="France is in Europe.", bm25_score=14.0, rank=2),
        ]
        self.retriever = MockRetriever(self.sample_passages)
        self.reader = MockReader(answer="Paris", qa_score=0.90, no_answer_prob=0.05)
        self.pipe = RAGAbstentionPipeline(
            retriever=self.retriever,
            reader=self.reader,
            bm25_threshold=15.0,
            classifier_threshold=0.30,
        )

    def test_empty_retrieval_handling(self):
        """When no passages are retrieved, pipeline should abstain cleanly without raising exceptions."""
        empty_retriever = MockRetriever(passages=[])
        pipe = RAGAbstentionPipeline(retriever=empty_retriever, reader=self.reader)
        res = pipe.predict("What is the capital of France?", top_k=5, strategy="always_answer")

        self.assertIsInstance(res, RAGResult)
        self.assertTrue(res.abstain)
        self.assertEqual(res.status, "ABSTAINED")
        self.assertIsNone(res.answer)
        self.assertEqual(len(res.retrieved_passages), 0)

    def test_always_answer_strategy(self):
        """Always answer baseline should accept non-empty answers regardless of threshold."""
        res = self.pipe.predict("What is the capital of France?", strategy="always_answer")
        self.assertEqual(res.status, "ANSWERED")
        self.assertEqual(res.answer, "Paris")
        self.assertFalse(res.abstain)

    def test_bm25_threshold_strategy_accept(self):
        """BM25 score 22.5 >= threshold 15.0 should accept answer."""
        res = self.pipe.predict("What is the capital of France?", strategy="bm25_threshold", threshold=15.0)
        self.assertEqual(res.status, "ANSWERED")
        self.assertEqual(res.answer, "Paris")
        self.assertFalse(res.abstain)

    def test_bm25_threshold_strategy_abstain(self):
        """BM25 score 22.5 < threshold 30.0 should abstain."""
        res = self.pipe.predict("What is the capital of France?", strategy="bm25_threshold", threshold=30.0)
        self.assertEqual(res.status, "ABSTAINED")
        self.assertIsNone(res.answer)
        self.assertTrue(res.abstain)

    def test_feature_consistency_with_model_if_present(self):
        """If answerability_logreg.joblib exists, its expected features must match FEATURES exactly."""
        model_path = Path(__file__).resolve().parents[1] / "models" / "answerability_logreg.joblib"
        if model_path.exists():
            model = joblib.load(model_path)
            # The model pipeline imputer or scaler learns feature names
            imputer = model.named_steps["imputer"]
            if hasattr(imputer, "feature_names_in_"):
                model_features = list(imputer.feature_names_in_)
                self.assertEqual(FEATURES, model_features)


if __name__ == "__main__":
    unittest.main()
