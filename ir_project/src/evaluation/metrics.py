"""
Information Retrieval Evaluation Metrics.

Implements standard IR metrics for measuring retrieval quality:
- Precision@k: Fraction of retrieved documents that are relevant
- Recall@k: Fraction of relevant documents that are retrieved
- MAP: Mean Average Precision across queries
- NDCG: Normalized Discounted Cumulative Gain (handles graded relevance)
- MRR: Mean Reciprocal Rank

All metrics follow standard IR evaluation conventions.
"""

from typing import List, Set, Dict, Union
import numpy as np


def precision_at_k(retrieved: List[str], 
                   relevant: Set[str], 
                   k: int) -> float:
    
    if k <= 0:
        return 0.0
    
    retrieved_at_k = retrieved[:k]
    relevant_retrieved = sum(1 for doc_id in retrieved_at_k if doc_id in relevant)
    
    return relevant_retrieved / k


def recall_at_k(retrieved: List[str], 
                relevant: Set[str], 
                k: int) -> float:
    
    if not relevant:
        return 0.0
    
    retrieved_at_k = retrieved[:k]
    relevant_retrieved = sum(1 for doc_id in retrieved_at_k if doc_id in relevant)
    
    return relevant_retrieved / len(relevant)


def f1_at_k(retrieved: List[str], 
            relevant: Set[str], 
            k: int) -> float:
    
    p = precision_at_k(retrieved, relevant, k)
    r = recall_at_k(retrieved, relevant, k)
    
    if p + r == 0:
        return 0.0
    
    return 2 * p * r / (p + r)


def average_precision(retrieved: List[str], 
                      relevant: Set[str]) -> float:
    
    if not relevant:
        return 0.0
    
    score = 0.0
    num_relevant_seen = 0
    
    for k, doc_id in enumerate(retrieved, 1):
        if doc_id in relevant:
            num_relevant_seen += 1
            precision = num_relevant_seen / k
            score += precision
    
    return score / len(relevant)


def mean_average_precision(all_retrieved: List[List[str]], 
                           all_relevant: List[Set[str]]) -> float:
    
    if not all_retrieved:
        return 0.0
    
    ap_scores = [
        average_precision(retrieved, relevant)
        for retrieved, relevant in zip(all_retrieved, all_relevant)
    ]
    
    return np.mean(ap_scores)


def dcg_at_k(relevances: List[float], k: int) -> float:
    
    relevances = np.array(relevances[:k])
    positions = np.arange(1, len(relevances) + 1)
    
    
    gains = (2 ** relevances - 1)
    discounts = np.log2(positions + 1)
    
    return np.sum(gains / discounts)


def ndcg_at_k(retrieved: List[str], 
              relevance_scores: Dict[str, float], 
              k: int) -> float:
    
    if k <= 0:
        return 0.0
    
    
    retrieved_relevances = [
        relevance_scores.get(doc_id, 0.0)
        for doc_id in retrieved[:k]
    ]
    
    
    dcg = dcg_at_k(retrieved_relevances, k)
    
    
    ideal_relevances = sorted(relevance_scores.values(), reverse=True)[:k]
    idcg = dcg_at_k(ideal_relevances, k)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def reciprocal_rank(retrieved: List[str], relevant: Set[str]) -> float:
    
    for rank, doc_id in enumerate(retrieved, 1):
        if doc_id in relevant:
            return 1.0 / rank
    
    return 0.0


def mean_reciprocal_rank(all_retrieved: List[List[str]], 
                         all_relevant: List[Set[str]]) -> float:
    
    if not all_retrieved:
        return 0.0
    
    rr_scores = [
        reciprocal_rank(retrieved, relevant)
        for retrieved, relevant in zip(all_retrieved, all_relevant)
    ]
    
    return np.mean(rr_scores)


def compute_all_metrics(retrieved: List[str],
                        relevant: Set[str],
                        k_values: List[int] = [1, 3, 5, 10],
                        relevance_scores: Dict[str, float] = None) -> Dict[str, float]:
    
    if relevance_scores is None:
        relevance_scores = {doc_id: 1.0 for doc_id in relevant}
    
    metrics = {}
    
    for k in k_values:
        metrics[f'P@{k}'] = precision_at_k(retrieved, relevant, k)
        metrics[f'R@{k}'] = recall_at_k(retrieved, relevant, k)
        metrics[f'F1@{k}'] = f1_at_k(retrieved, relevant, k)
        metrics[f'NDCG@{k}'] = ndcg_at_k(retrieved, relevance_scores, k)
    
    metrics['AP'] = average_precision(retrieved, relevant)
    metrics['RR'] = reciprocal_rank(retrieved, relevant)
    
    return metrics
