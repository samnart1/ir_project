"""
Retrieval methods for the IR system.

Implements multiple retrieval strategies:
- TF-IDF: Term Frequency-Inverse Document Frequency
- BM25: Best Matching 25 (Okapi BM25)
- Dense: Neural embedding-based retrieval
- Hybrid: Combination of sparse and dense methods
"""

from .base import BaseRetriever, RetrievalResult
from .tfidf import TFIDFRetriever
from .bm25 import BM25Retriever
from .dense import DenseRetriever
from .hybrid import HybridRetriever

__all__ = [
    'BaseRetriever',
    'RetrievalResult', 
    'TFIDFRetriever',
    'BM25Retriever',
    'DenseRetriever',
    'HybridRetriever'
]
