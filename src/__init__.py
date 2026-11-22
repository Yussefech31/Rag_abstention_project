"""
RAG Abstention Package
Knowing When Not to Answer: Abstention in Retrieval-Augmented Question Answering
"""

from src.pipeline import (
    RAGAbstentionPipeline,
    RAGResult,
    Passage,
    BM25Retriever,
    QAReader,
    create_pipeline,
    FEATURES,
)

__all__ = [
    "RAGAbstentionPipeline",
    "RAGResult",
    "Passage",
    "BM25Retriever",
    "QAReader",
    "create_pipeline",
    "FEATURES",
]
