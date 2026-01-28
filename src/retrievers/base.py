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
    """Represents a single retrieval result."""
    document: Document
    score: float
    rank: int
    
    def __repr__(self):
        return f"RetrievalResult(doc_id={self.document.doc_id}, score={self.score:.4f}, rank={self.rank})"


class BaseRetriever(ABC):
    """Abstract base class for all retrievers."""
    
    def __init__(self, corpus: Optional[Corpus] = None):
        self.corpus = corpus
        self._is_indexed = False
    
    @property
    def name(self) -> str:
        """Return the name of the retriever."""
        return self.__class__.__name__
    
    @abstractmethod
    def index(self, corpus: Corpus) -> None:
        """
        Build the index from the corpus.
        
        Args:
            corpus: The document corpus to index
        """
        pass
    
    @abstractmethod
    def _retrieve_scores(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """
        Internal method to retrieve document indices and scores.
        
        Args:
            query: The search query
            top_k: Number of results to return
            
        Returns:
            List of (document_index, score) tuples
        """
        pass
    
    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: The search query
            top_k: Number of results to return
            
        Returns:
            List of RetrievalResult objects
        """
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
        """
        Retrieve document IDs for a query.
        
        Args:
            query: The search query
            top_k: Number of results to return
            
        Returns:
            List of document IDs
        """
        results = self.retrieve(query, top_k)
        return [r.document.doc_id for r in results]
