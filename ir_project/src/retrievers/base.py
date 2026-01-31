"""
Base retriever interface.

Defines the common interface that all retrieval methods must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional
from ..corpus import Corpus, Document


@dataclass
class RetrievalResult:
    
    document: Document
    score: float
    rank: int
    
    def __repr__(self):
        return f"RetrievalResult(doc_id={self.document.doc_id}, score={self.score:.4f}, rank={self.rank})"


class BaseRetriever(ABC):
    
    def __init__(self, corpus: Optional[Corpus] = None):
        self.corpus = corpus
        self._is_indexed = False
    
    @property
    def name(self) -> str:
        return self.__class__.__name__
    
    @abstractmethod
    def index(self, corpus: Corpus) -> None:
        
        pass
    
    @abstractmethod
    def _retrieve_scores(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        
        pass
    
    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        
        if not self._is_indexed:
            raise RuntimeError(f"{self.name} has not been indexed. Call index() first.")
        
        results = self._retrieve_scores(query, top_k)
        
        return [
            RetrievalResult(
                document=self.corpus.get_by_index(idx),
                score=score,
                rank=rank + 1
            )
            for rank, (idx, score) in enumerate(results)
        ]
    
    def retrieve_ids(self, query: str, top_k: int = 10) -> List[str]:
        
        results = self.retrieve(query, top_k)
        return [r.document.doc_id for r in results]
