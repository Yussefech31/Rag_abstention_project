"""
src/pipeline.py
End-to-End Inference Pipeline for Selective Prediction & Abstention in RAG.

Stages:
1. Lexical Retrieval (BM25Okapi over indexed passage corpus)
2. Neural Reading (deepset/roberta-base-squad2 extractive span prediction)
3. 11-Dimensional Feature Extraction (retrieval dynamics + reader uncertainty)
4. Decision Engine (Classifier gating with threshold tau = 0.28, BM25 threshold, or Always Answer)
"""

import os
import re
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
import joblib

# ── Logging Configuration ──────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RAGPipeline")

# ── Default Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "answerability_logreg.joblib"
DEFAULT_TEST_SPLIT = PROJECT_ROOT / "data" / "splits" / "test.csv"
DEFAULT_PASSAGES_CLEAN = PROJECT_ROOT / "data" / "processed" / "passages_clean.csv"

# ── Feature Specification (strictly aligned with 10 inference-available features) ──
FEATURES = [
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

# ── Tokenization Helpers ───────────────────────────────────────────────────────
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
TOKEN_SET_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric words."""
    return TOKEN_PATTERN.findall(str(text).lower())


def token_set(text: str) -> set:
    """Extract set of unique alphanumeric lowercase tokens."""
    return set(TOKEN_SET_RE.findall(str(text).lower()))


# ── Data Classes ───────────────────────────────────────────────────────────────
@dataclass
class Passage:
    paragraph_id: str
    article_title: str
    context: str
    bm25_score: float = 0.0
    rank: int = 0


@dataclass
class RAGResult:
    question: str
    answer: Optional[str]
    abstain: bool
    status: str  # "ANSWERED" | "ABSTAINED"
    strategy: str  # "classifier" | "bm25_threshold" | "always_answer"
    decision_probability: float  # P(is_answerable)
    threshold: float
    qa_score: float
    qa_no_answer_prob: float
    top_bm25_score: float
    best_passage: Optional[Dict[str, Any]]
    retrieved_passages: List[Dict[str, Any]]
    features: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── BM25 Retriever ─────────────────────────────────────────────────────────────
class BM25Retriever:
    """Sparse lexical retriever powered by Rank-BM25 (BM25Okapi)."""

    def __init__(self, corpus_df: Optional[pd.DataFrame] = None):
        """
        Initialize the retriever with a passage DataFrame containing:
        'paragraph_id', 'article_title', 'context'.
        """
        self.corpus: Optional[pd.DataFrame] = None
        self.index = None
        if corpus_df is not None:
            self.fit(corpus_df)

    def fit(self, corpus_df: pd.DataFrame):
        """Build the BM25 inverted index from a DataFrame of passages."""
        from rank_bm25 import BM25Okapi

        required = {"paragraph_id", "article_title", "context"}
        if not required.issubset(corpus_df.columns):
            raise ValueError(f"Corpus DataFrame must contain columns: {required}")

        self.corpus = (
            corpus_df[["paragraph_id", "article_title", "context"]]
            .drop_duplicates("paragraph_id")
            .reset_index(drop=True)
        )
        logger.info("Indexing %d unique passages with BM25Okapi...", len(self.corpus))
        tokenized_corpus = [tokenize(doc) for doc in self.corpus["context"]]
        self.index = BM25Okapi(tokenized_corpus)
        logger.info("BM25 index built successfully.")

    @classmethod
    def from_csv(cls, path: Union[str, Path]) -> "BM25Retriever":
        """Load passages from CSV and build index."""
        df = pd.read_csv(path)
        return cls(df)

    def retrieve(self, question: str, top_k: int = 5) -> List[Passage]:
        """Retrieve top-k ranked passages for a given question string."""
        if self.index is None or self.corpus is None:
            raise RuntimeError("Retriever has not been fitted with a corpus.")

        q_tokens = tokenize(question)
        if not q_tokens:
            return []

        scores = self.index.get_scores(q_tokens)
        order = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank_idx, idx in enumerate(order, start=1):
            row = self.corpus.iloc[idx]
            results.append(
                Passage(
                    paragraph_id=str(row["paragraph_id"]),
                    article_title=str(row["article_title"]),
                    context=str(row["context"]),
                    bm25_score=float(scores[idx]),
                    rank=rank_idx,
                )
            )
        return results


# ── Extractive QA Reader ───────────────────────────────────────────────────────
class QAReader:
    """
    Extractive Question Answering Reader using deepset/roberta-base-squad2.
    Supports lazy model loading to minimize memory footprint until requested.
    """

    def __init__(self, model_name: str = "deepset/roberta-base-squad2", device: Optional[int] = None):
        self.model_name = model_name
        self.device = device
        self._pipeline = None

    def _load_pipeline(self):
        if self._pipeline is None:
            import torch
            from transformers import pipeline

            if self.device is None:
                self.device = 0 if torch.cuda.is_available() else -1

            logger.info("Loading QA pipeline: %s (device=%d)...", self.model_name, self.device)
            self._pipeline = pipeline(
                "question-answering",
                model=self.model_name,
                tokenizer=self.model_name,
                device=self.device,
            )
            logger.info("QA Reader ready.")

    def extract_answer(self, question: str, context: str) -> Dict[str, Any]:
        """Extract answer span from single context string."""
        self._load_pipeline()
        try:
            pred = self._pipeline(
                question=question,
                context=context,
                handle_impossible_answer=True,
            )
            return {
                "answer": str(pred.get("answer", "")).strip(),
                "qa_score": float(pred.get("score", 0.0)),
                "qa_no_answer_probability": float(pred.get("no_answer_probability", 0.0)),
            }
        except Exception as e:
            logger.warning("Error running QA Reader on context: %s", e)
            return {
                "answer": "",
                "qa_score": 0.0,
                "qa_no_answer_probability": 1.0,
            }

    def read_passages(self, question: str, passages: List[Passage]) -> Tuple[Dict[str, Any], Optional[Passage]]:
        """
        Evaluate question against all candidate passages and select the best answer.
        Returns (best_pred_dict, best_passage).
        """
        if not passages:
            return {
                "answer": "",
                "qa_score": 0.0,
                "qa_no_answer_probability": 1.0,
            }, None

        candidates = []
        for p in passages:
            res = self.extract_answer(question, p.context)
            candidates.append((res, p))

        # Select candidate with highest qa_score
        best_res, best_p = max(candidates, key=lambda cp: cp[0]["qa_score"])
        return best_res, best_p


# ── Full RAG Pipeline with Selective Abstention ───────────────────────────────
class RAGAbstentionPipeline:
    """
    Complete Retrieval-Augmented QA Pipeline equipped with Selective Prediction.

    Supports three decision strategies:
    - 'classifier': Logistic Regression over 10 features with validation-tuned threshold tau = 0.42
    - 'bm25_threshold': Threshold heuristic on top BM25 retrieval score (tau = 13.15)
    - 'always_answer': Baseline without abstention mechanism
    """

    DEFAULT_CLASSIFIER_THRESHOLD = 0.42
    DEFAULT_BM25_THRESHOLD = 13.15

    def __init__(
        self,
        retriever: BM25Retriever,
        reader: Optional[QAReader] = None,
        model_path: Optional[Union[str, Path]] = None,
        default_strategy: str = "classifier",
        classifier_threshold: float = DEFAULT_CLASSIFIER_THRESHOLD,
        bm25_threshold: float = DEFAULT_BM25_THRESHOLD,
    ):
        self.retriever = retriever
        self.reader = reader or QAReader()
        self.default_strategy = default_strategy
        self.classifier_threshold = classifier_threshold
        self.bm25_threshold = bm25_threshold

        # Load trained classifier pipeline
        self.model_path = Path(model_path or DEFAULT_MODEL_PATH)
        self.classifier = None
        self._load_classifier()

    def _load_classifier(self):
        if self.model_path.exists():
            logger.info("Loading answerability classifier from %s", self.model_path)
            self.classifier = joblib.load(self.model_path)
        else:
            logger.warning(
                "Classifier artifact not found at %s. Classifier strategy will be unavailable until trained.",
                self.model_path,
            )

    def extract_features(
        self,
        question: str,
        retrieved_passages: List[Passage],
        qa_pred: Dict[str, Any],
        gold_passage_id: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Extract the 10 inference-available features required by the Answerability Classifier.
        (source_passage_retrieved is excluded from the predictive feature set).
        """
        # Retrieval features
        if retrieved_passages:
            scores = [p.bm25_score for p in retrieved_passages]
            top_bm25_score = float(scores[0])
            second_bm25_score = float(scores[1]) if len(scores) > 1 else 0.0
            bm25_score_gap = top_bm25_score - second_bm25_score
            mean_topk_bm25 = float(np.mean(scores))
        else:
            top_bm25_score = 0.0
            second_bm25_score = 0.0
            bm25_score_gap = 0.0
            mean_topk_bm25 = 0.0

        # Query & Reader features
        q_text = str(question or "")
        pred_ans = str(qa_pred.get("answer", "") or "").strip()

        question_length = float(len(q_text.split()))
        predicted_answer_length = float(len(pred_ans.split()))
        qa_score = float(qa_pred.get("qa_score", 0.0))
        qa_no_answer_prob = float(qa_pred.get("qa_no_answer_probability", 0.0))
        question_token_count = float(len(token_set(q_text)))
        answer_nonempty = 1.0 if len(pred_ans) > 0 else 0.0

        feat_dict = {
            "top_bm25_score": top_bm25_score,
            "second_bm25_score": second_bm25_score,
            "bm25_score_gap": bm25_score_gap,
            "mean_topk_bm25": mean_topk_bm25,
            "qa_score": qa_score,
            "qa_no_answer_probability": qa_no_answer_prob,
            "predicted_answer_length": predicted_answer_length,
            "answer_nonempty": answer_nonempty,
            "question_length": question_length,
            "question_token_count": question_token_count,
        }

        # Build single-row DataFrame with strict column order
        feat_df = pd.DataFrame([feat_dict], columns=FEATURES)
        return feat_df, feat_dict

    def predict(
        self,
        question: str,
        top_k: int = 5,
        strategy: Optional[str] = None,
        threshold: Optional[float] = None,
        gold_passage_id: Optional[str] = None,
    ) -> RAGResult:
        """
        Execute full end-to-end question answering with selective abstention.

        Parameters:
        -----------
        question: Input user question
        top_k: Number of passages to retrieve with BM25
        strategy: 'classifier', 'bm25_threshold', or 'always_answer'
        threshold: Custom threshold for classifier or BM25 gating
        gold_passage_id: Optional ground-truth passage identifier for diagnostics
        """
        strat = (strategy or self.default_strategy).lower()

        # Step 1: BM25 Passage Retrieval
        passages = self.retriever.retrieve(question, top_k=top_k)

        # Step 2: Extractive QA Reading
        qa_pred, best_p = self.reader.read_passages(question, passages)

        # Step 3: Feature Extraction
        feat_df, feat_dict = self.extract_features(
            question=question,
            retrieved_passages=passages,
            qa_pred=qa_pred,
            gold_passage_id=gold_passage_id,
        )

        # Step 4: Decision Strategy
        prob = 1.0
        abstain = False
        applied_threshold = 0.0

        if strat == "always_answer":
            abstain = False
            applied_threshold = 0.0
            prob = 1.0

        elif strat == "bm25_threshold":
            applied_threshold = threshold if threshold is not None else self.bm25_threshold
            top_score = feat_dict["top_bm25_score"]
            abstain = top_score < applied_threshold
            prob = float(top_score)

        elif strat == "classifier":
            if self.classifier is None:
                raise RuntimeError("Classifier model is not loaded. Train or specify valid model_path.")
            applied_threshold = threshold if threshold is not None else self.classifier_threshold
            prob = float(self.classifier.predict_proba(feat_df)[:, 1][0])
            abstain = prob < applied_threshold

        else:
            raise ValueError(f"Unknown strategy '{strat}'. Use 'classifier', 'bm25_threshold', or 'always_answer'.")

        raw_answer = qa_pred.get("answer", "")
        # Also treat empty model answers as abstained if classifier accepted but span is empty
        final_answer = raw_answer if (not abstain and raw_answer) else None
        status = "ABSTAINED" if abstain or not final_answer else "ANSWERED"

        return RAGResult(
            question=question,
            answer=final_answer,
            abstain=(status == "ABSTAINED"),
            status=status,
            strategy=strat,
            decision_probability=prob,
            threshold=applied_threshold,
            qa_score=feat_dict["qa_score"],
            qa_no_answer_prob=feat_dict["qa_no_answer_probability"],
            top_bm25_score=feat_dict["top_bm25_score"],
            best_passage=asdict(best_p) if best_p else None,
            retrieved_passages=[asdict(p) for p in passages],
            features=feat_dict,
        )


# ── Convenience Pipeline Factory ───────────────────────────────────────────────
def create_pipeline(
    corpus_source: Optional[Union[str, Path, pd.DataFrame]] = None,
    model_path: Optional[Union[str, Path]] = None,
    lazy_reader: bool = True,
) -> RAGAbstentionPipeline:
    """
    Convenience factory to instantiate a ready-to-use RAGAbstentionPipeline.
    Defaults to loading passages from data/splits/test.csv if no corpus is specified.
    """
    if corpus_source is None:
        if DEFAULT_TEST_SPLIT.exists():
            logger.info("Loading default corpus from %s", DEFAULT_TEST_SPLIT)
            corpus_df = pd.read_csv(DEFAULT_TEST_SPLIT)
        elif DEFAULT_PASSAGES_CLEAN.exists():
            logger.info("Loading default corpus from %s", DEFAULT_PASSAGES_CLEAN)
            corpus_df = pd.read_csv(DEFAULT_PASSAGES_CLEAN)
        else:
            raise FileNotFoundError("Could not locate default corpus dataset in data/.")
    elif isinstance(corpus_source, (str, Path)):
        corpus_df = pd.read_csv(corpus_source)
    else:
        corpus_df = corpus_source

    retriever = BM25Retriever(corpus_df)
    reader = QAReader()
    if not lazy_reader:
        reader._load_pipeline()

    return RAGAbstentionPipeline(
        retriever=retriever,
        reader=reader,
        model_path=model_path or DEFAULT_MODEL_PATH,
    )


# ── Quick CLI Demonstration ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print(" RAG Abstention Pipeline — Standalone Test Run")
    print("=" * 70)

    try:
        pipeline = create_pipeline()
    except Exception as err:
        print(f"Error initializing pipeline: {err}")
        exit(1)

    test_queries = [
        "In what year did the team win the championship?",
        "What is the capital of France?",
        "Who was the prime minister of Atlantis in 1842?",  # Adversarial / unanswerable
    ]

    print("\nRunning inference on sample queries:\n")
    for q in test_queries:
        print("-" * 70)
        print(f"Query: '{q}'")
        res = pipeline.predict(q, top_k=5, strategy="classifier")
        print(f"Status:       {res.status}")
        print(f"Decision:     {'ABSTAIN (∅)' if res.abstain else res.answer}")
        print(f"P(Answerable): {res.decision_probability:.4f} (Threshold: {res.threshold})")
        print(f"Top BM25:     {res.top_bm25_score:.2f}")
        print(f"QA Score:     {res.qa_score:.4f} | No-Answer Prob: {res.qa_no_answer_prob:.4f}")
        if res.best_passage:
            print(f"Top Passage:  [{res.best_passage['article_title']}] {res.best_passage['context'][:120]}...")
    print("=" * 70)
