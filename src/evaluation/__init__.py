"""
Evaluation module for Information Retrieval.

Implements standard IR evaluation metrics and benchmarking utilities.
"""

from .metrics import (
    precision_at_k,
    recall_at_k,
    average_precision,
    mean_average_precision,
    ndcg_at_k,
    reciprocal_rank,
    mean_reciprocal_rank,
    f1_at_k
)

from .benchmark import Benchmark, QueryRelevance

__all__ = [
    'precision_at_k',
    'recall_at_k',
    'average_precision',
    'mean_average_precision',
    'ndcg_at_k',
    'reciprocal_rank',
    'mean_reciprocal_rank',
    'f1_at_k',
    'Benchmark',
    'QueryRelevance'
]
