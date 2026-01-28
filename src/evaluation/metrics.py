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
    """
    Calculate Precision@k.
    
    Precision@k = |retrieved[:k] ∩ relevant| / k
    
    Measures the fraction of top-k retrieved documents that are relevant.
    
    Args:
        retrieved: Ranked list of retrieved document IDs
        relevant: Set of relevant document IDs
        k: Cutoff position
        
    Returns:
        Precision@k score (0.0 to 1.0)
    """
    if k <= 0:
        return 0.0
    
    retrieved_at_k = retrieved[:k]
    relevant_retrieved = sum(1 for doc_id in retrieved_at_k if doc_id in relevant)
    
    return relevant_retrieved / k


def recall_at_k(retrieved: List[str], 
                relevant: Set[str], 
                k: int) -> float:
    """
    Calculate Recall@k.
    
    Recall@k = |retrieved[:k] ∩ relevant| / |relevant|
    
    Measures the fraction of relevant documents found in top-k.
    
    Args:
        retrieved: Ranked list of retrieved document IDs
        relevant: Set of relevant document IDs
        k: Cutoff position
        
    Returns:
        Recall@k score (0.0 to 1.0)
    """
    if not relevant:
        return 0.0
    
    retrieved_at_k = retrieved[:k]
    relevant_retrieved = sum(1 for doc_id in retrieved_at_k if doc_id in relevant)
    
    return relevant_retrieved / len(relevant)


def f1_at_k(retrieved: List[str], 
            relevant: Set[str], 
            k: int) -> float:
    """
    Calculate F1@k (harmonic mean of precision and recall at k).
    
    Args:
        retrieved: Ranked list of retrieved document IDs
        relevant: Set of relevant document IDs
        k: Cutoff position
        
    Returns:
        F1@k score (0.0 to 1.0)
    """
    p = precision_at_k(retrieved, relevant, k)
    r = recall_at_k(retrieved, relevant, k)
    
    if p + r == 0:
        return 0.0
    
    return 2 * p * r / (p + r)


def average_precision(retrieved: List[str], 
                      relevant: Set[str]) -> float:
    """
    Calculate Average Precision (AP).
    
    AP = (1/|relevant|) * Σ (precision@k * rel(k))
    
    where rel(k) = 1 if document at position k is relevant, 0 otherwise.
    
    Average Precision considers both precision and the rank of relevant documents.
    It rewards systems that rank relevant documents higher.
    
    Args:
        retrieved: Ranked list of retrieved document IDs
        relevant: Set of relevant document IDs
        
    Returns:
        Average Precision score (0.0 to 1.0)
    """
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
    """
    Calculate Mean Average Precision (MAP).
    
    MAP = (1/|Q|) * Σ AP(q) for all queries q in Q
    
    MAP is the primary metric for evaluating ranked retrieval systems.
    
    Args:
        all_retrieved: List of retrieved document lists (one per query)
        all_relevant: List of relevant document sets (one per query)
        
    Returns:
        MAP score (0.0 to 1.0)
    """
    if not all_retrieved:
        return 0.0
    
    ap_scores = [
        average_precision(retrieved, relevant)
        for retrieved, relevant in zip(all_retrieved, all_relevant)
    ]
    
    return np.mean(ap_scores)


def dcg_at_k(relevances: List[float], k: int) -> float:
    """
    Calculate Discounted Cumulative Gain at k.
    
    DCG@k = Σ (2^rel_i - 1) / log2(i + 1) for i = 1 to k
    
    Args:
        relevances: List of relevance scores in ranked order
        k: Cutoff position
        
    Returns:
        DCG@k score
    """
    relevances = np.array(relevances[:k])
    positions = np.arange(1, len(relevances) + 1)
    
    # Using the standard DCG formula
    gains = (2 ** relevances - 1)
    discounts = np.log2(positions + 1)
    
    return np.sum(gains / discounts)


def ndcg_at_k(retrieved: List[str], 
              relevance_scores: Dict[str, float], 
              k: int) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at k (NDCG@k).
    
    NDCG@k = DCG@k / IDCG@k
    
    NDCG handles graded relevance (not just binary) and normalizes
    by the ideal ranking, giving scores in [0, 1].
    
    Args:
        retrieved: Ranked list of retrieved document IDs
        relevance_scores: Dict mapping doc_id to relevance score (e.g., 0, 1, 2, 3)
        k: Cutoff position
        
    Returns:
        NDCG@k score (0.0 to 1.0)
    """
    if k <= 0:
        return 0.0
    
    # Get relevance scores for retrieved documents
    retrieved_relevances = [
        relevance_scores.get(doc_id, 0.0)
        for doc_id in retrieved[:k]
    ]
    
    # Calculate DCG
    dcg = dcg_at_k(retrieved_relevances, k)
    
    # Calculate ideal DCG (best possible ranking)
    ideal_relevances = sorted(relevance_scores.values(), reverse=True)[:k]
    idcg = dcg_at_k(ideal_relevances, k)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def reciprocal_rank(retrieved: List[str], relevant: Set[str]) -> float:
    """
    Calculate Reciprocal Rank.
    
    RR = 1 / rank of first relevant document
    
    Args:
        retrieved: Ranked list of retrieved document IDs
        relevant: Set of relevant document IDs
        
    Returns:
        Reciprocal Rank score (0.0 to 1.0)
    """
    for rank, doc_id in enumerate(retrieved, 1):
        if doc_id in relevant:
            return 1.0 / rank
    
    return 0.0


def mean_reciprocal_rank(all_retrieved: List[List[str]], 
                         all_relevant: List[Set[str]]) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR).
    
    MRR = (1/|Q|) * Σ RR(q) for all queries q in Q
    
    MRR is useful when there's typically one "correct" answer.
    
    Args:
        all_retrieved: List of retrieved document lists (one per query)
        all_relevant: List of relevant document sets (one per query)
        
    Returns:
        MRR score (0.0 to 1.0)
    """
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
    """
    Compute all metrics for a single query.
    
    Args:
        retrieved: Ranked list of retrieved document IDs
        relevant: Set of relevant document IDs
        k_values: List of k values for P@k, R@k, NDCG@k
        relevance_scores: Optional graded relevance scores for NDCG
        
    Returns:
        Dictionary of metric names to scores
    """
    # Use binary relevance for NDCG if no grades provided
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
