"""
Hybrid Retriever combining multiple retrieval methods.

Implements reciprocal rank fusion and weighted score combination
to leverage the strengths of both sparse (BM25) and dense (neural)
retrieval methods.
"""

from typing import List, Tuple, Dict, Optional
from collections import defaultdict

from .base import BaseRetriever, RetrievalResult
from .bm25 import BM25Retriever
from .dense import DenseRetriever
from ..corpus import Corpus


class HybridRetriever(BaseRetriever):
    
    
    def __init__(self,
                 sparse_retriever: Optional[BM25Retriever] = None,
                 dense_retriever: Optional[DenseRetriever] = None,
                 fusion_method: str = "rrf",
                 sparse_weight: float = 0.5,
                 rrf_k: int = 60):
        
        super().__init__()
        self.sparse_retriever = sparse_retriever or BM25Retriever()
        self.dense_retriever = dense_retriever or DenseRetriever()
        self.fusion_method = fusion_method
        self.sparse_weight = sparse_weight
        self.dense_weight = 1.0 - sparse_weight
        self.rrf_k = rrf_k
    
    def index(self, corpus: Corpus) -> None:
        
        self.corpus = corpus
        
        print("[Hybrid] Building sparse index...")
        self.sparse_retriever.index(corpus)
        
        print("[Hybrid] Building dense index...")
        self.dense_retriever.index(corpus)
        
        self._is_indexed = True
        print(f"[Hybrid] Indexed {len(corpus)} documents with both methods")
    
    def _reciprocal_rank_fusion(self, 
                                 sparse_results: List[Tuple[int, float]],
                                 dense_results: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
        
        rrf_scores: Dict[int, float] = defaultdict(float)
        
        
        for rank, (doc_idx, _) in enumerate(sparse_results):
            rrf_scores[doc_idx] += 1.0 / (self.rrf_k + rank + 1)
        
        
        for rank, (doc_idx, _) in enumerate(dense_results):
            rrf_scores[doc_idx] += 1.0 / (self.rrf_k + rank + 1)
        
        
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_results
    
    def _weighted_combination(self,
                               sparse_results: List[Tuple[int, float]],
                               dense_results: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
        
        combined_scores: Dict[int, float] = defaultdict(float)
        
        
        if sparse_results:
            sparse_scores = [score for _, score in sparse_results]
            max_sparse = max(sparse_scores) if sparse_scores else 1.0
            min_sparse = min(sparse_scores) if sparse_scores else 0.0
            range_sparse = max_sparse - min_sparse if max_sparse != min_sparse else 1.0
            
            for doc_idx, score in sparse_results:
                normalized = (score - min_sparse) / range_sparse
                combined_scores[doc_idx] += self.sparse_weight * normalized
        
        
        if dense_results:
            dense_scores = [score for _, score in dense_results]
            max_dense = max(dense_scores) if dense_scores else 1.0
            min_dense = min(dense_scores) if dense_scores else 0.0
            range_dense = max_dense - min_dense if max_dense != min_dense else 1.0
            
            for doc_idx, score in dense_results:
                normalized = (score - min_dense) / range_dense
                combined_scores[doc_idx] += self.dense_weight * normalized
        
        
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_results
    
    def _retrieve_scores(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        
        fetch_k = min(top_k * 3, len(self.corpus))
        
        sparse_results = self.sparse_retriever._retrieve_scores(query, fetch_k)
        dense_results = self.dense_retriever._retrieve_scores(query, fetch_k)
        
        if self.fusion_method == "rrf":
            combined = self._reciprocal_rank_fusion(sparse_results, dense_results)
        else:
            combined = self._weighted_combination(sparse_results, dense_results)
        
        return combined[:top_k]
    
    def retrieve_with_breakdown(self, query: str, top_k: int = 10) -> Dict:
        
        sparse_results = self.sparse_retriever.retrieve(query, top_k)
        dense_results = self.dense_retriever.retrieve(query, top_k)
        hybrid_results = self.retrieve(query, top_k)
        
        return {
            "sparse": sparse_results,
            "dense": dense_results,
            "hybrid": hybrid_results,
            "sparse_ids": [r.document.doc_id for r in sparse_results],
            "dense_ids": [r.document.doc_id for r in dense_results],
            "hybrid_ids": [r.document.doc_id for r in hybrid_results]
        }
