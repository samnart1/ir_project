"""
Benchmarking suite for comparing retrieval methods.

Provides utilities for:
- Loading and managing relevance judgments
- Running systematic evaluations
- Generating comparison reports
"""

import json
import time
from dataclasses import dataclass, field
from typing import List, Set, Dict, Optional
from pathlib import Path
import numpy as np

from .metrics import (
    precision_at_k, recall_at_k, f1_at_k,
    average_precision, mean_average_precision,
    ndcg_at_k, mean_reciprocal_rank, reciprocal_rank
)
from ..retrievers.base import BaseRetriever


@dataclass
class QueryRelevance:
    query_id: str
    query_text: str
    relevant_docs: Set[str] = field(default_factory=set)
    graded_relevance: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "query_id": self.query_id,
            "query_text": self.query_text,
            "relevant_docs": list(self.relevant_docs),
            "graded_relevance": self.graded_relevance
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "QueryRelevance":
        return cls(
            query_id=data["query_id"],
            query_text=data["query_text"],
            relevant_docs=set(data.get("relevant_docs", [])),
            graded_relevance=data.get("graded_relevance", {})
        )


class Benchmark:
    
    def __init__(self, 
                 queries: Optional[List[QueryRelevance]] = None,
                 k_values: List[int] = [1, 3, 5, 10]):
    
        self.queries = queries or []
        self.k_values = k_values
    
    def add_query(self, query: QueryRelevance):
        
        self.queries.append(query)
    
    def evaluate(self, 
                 retriever: BaseRetriever,
                 top_k: int = 100,
                 verbose: bool = True) -> Dict[str, float]:
        
        if not self.queries:
            raise ValueError("No queries in benchmark. Add queries first.")
        
        all_retrieved = []
        all_relevant = []
        all_metrics = []
        total_time = 0.0
        
        for query in self.queries:
            start = time.time()
            retrieved_ids = retriever.retrieve_ids(query.query_text, top_k)
            elapsed = time.time() - start
            total_time += elapsed
            
            all_retrieved.append(retrieved_ids)
            all_relevant.append(query.relevant_docs)
            
            query_metrics = {}
            for k in self.k_values:
                query_metrics[f'P@{k}'] = precision_at_k(retrieved_ids, query.relevant_docs, k)
                query_metrics[f'R@{k}'] = recall_at_k(retrieved_ids, query.relevant_docs, k)
                query_metrics[f'F1@{k}'] = f1_at_k(retrieved_ids, query.relevant_docs, k)
                
                rel_scores = query.graded_relevance if query.graded_relevance else {d: 1.0 for d in query.relevant_docs}
                query_metrics[f'NDCG@{k}'] = ndcg_at_k(retrieved_ids, rel_scores, k)
            
            query_metrics['AP'] = average_precision(retrieved_ids, query.relevant_docs)
            query_metrics['RR'] = reciprocal_rank(retrieved_ids, query.relevant_docs)
            
            all_metrics.append(query_metrics)
        
        aggregated = {}
        metric_names = all_metrics[0].keys()
        
        for metric in metric_names:
            values = [m[metric] for m in all_metrics]
            aggregated[metric] = np.mean(values)
            aggregated[f'{metric}_std'] = np.std(values)
        
        aggregated['MAP'] = mean_average_precision(all_retrieved, all_relevant)
        aggregated['MRR'] = mean_reciprocal_rank(all_retrieved, all_relevant)
        
        aggregated['avg_query_time_ms'] = (total_time / len(self.queries)) * 1000
        aggregated['total_time_s'] = total_time
        
        if verbose:
            self._print_results(retriever.name, aggregated)
        
        return aggregated
    
    def compare(self,
                retrievers: List[BaseRetriever],
                top_k: int = 100) -> Dict[str, Dict[str, float]]:
        
        print("=" * 70)
        print("RETRIEVER COMPARISON")
        print("=" * 70)
        print(f"Queries: {len(self.queries)}")
        print(f"K values: {self.k_values}")
        print("=" * 70)
        
        results = {}
        for retriever in retrievers:
            print(f"\nEvaluating: {retriever.name}")
            print("-" * 40)
            results[retriever.name] = self.evaluate(retriever, top_k, verbose=True)
        
        self._print_comparison_table(results)
        
        return results
    
    def _print_results(self, name: str, metrics: Dict[str, float]):
        print(f"\n{name} Results:")
        print("-" * 40)
        
        key_metrics = ['MAP', 'MRR', 'P@10', 'R@10', 'NDCG@10']
        for metric in key_metrics:
            if metric in metrics:
                print(f"  {metric:12s}: {metrics[metric]:.4f}")
        
        print(f"  {'Avg Time':12s}: {metrics['avg_query_time_ms']:.2f} ms/query")
    
    def _print_comparison_table(self, results: Dict[str, Dict[str, float]]):
        print("\n" + "=" * 70)
        print("COMPARISON SUMMARY")
        print("=" * 70)
        
        metrics_to_show = ['MAP', 'MRR', 'P@5', 'P@10', 'R@10', 'NDCG@10']
        
        
        header = f"{'Retriever':20s}"
        for metric in metrics_to_show:
            header += f" {metric:>10s}"
        header += f" {'Time(ms)':>10s}"
        print(header)
        print("-" * 70)
        
        
        for name, metrics in results.items():
            row = f"{name:20s}"
            for metric in metrics_to_show:
                row += f" {metrics.get(metric, 0):>10.4f}"
            row += f" {metrics['avg_query_time_ms']:>10.2f}"
            print(row)
        
        print("=" * 70)
        
        print("\nBest performers:")
        for metric in metrics_to_show:
            best_name = max(results.keys(), key=lambda n: results[n].get(metric, 0))
            best_value = results[best_name].get(metric, 0)
            print(f"  {metric}: {best_name} ({best_value:.4f})")
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump([q.to_dict() for q in self.queries], f, indent=2)
    
    @classmethod
    def load(cls, path: str, k_values: List[int] = [1, 3, 5, 10]) -> "Benchmark":
        with open(path, 'r') as f:
            data = json.load(f)
        queries = [QueryRelevance.from_dict(q) for q in data]
        return cls(queries=queries, k_values=k_values)


def create_sample_benchmark() -> Benchmark:
    
    queries = [
        QueryRelevance(
            query_id="q1",
            query_text="neural network deep learning",
            relevant_docs={"doc1", "doc3", "doc7"},
            graded_relevance={"doc1": 3, "doc3": 2, "doc7": 1}
        ),
        QueryRelevance(
            query_id="q2", 
            query_text="natural language processing transformers",
            relevant_docs={"doc2", "doc4", "doc8"},
            graded_relevance={"doc2": 3, "doc4": 2, "doc8": 2}
        ),
    ]
    return Benchmark(queries=queries)
