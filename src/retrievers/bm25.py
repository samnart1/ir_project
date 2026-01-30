"""
BM25 (Best Matching 25) Retriever.

Implements the Okapi BM25 ranking function, which is a probabilistic
retrieval model that extends TF-IDF with document length normalization
and term frequency saturation.
"""

from typing import List, Tuple
import numpy as np
from rank_bm25 import BM25Okapi

from .base import BaseRetriever
from ..corpus import Corpus, TextPreprocessor


class BM25Retriever(BaseRetriever):
    
    
    def __init__(self, 
                 k1: float = 1.5,
                 b: float = 0.75,
                 use_preprocessing: bool = True):
        
        super().__init__()
        self.k1 = k1
        self.b = b
        self.use_preprocessing = use_preprocessing
        
        self.preprocessor = TextPreprocessor() if use_preprocessing else None
        self.bm25 = None
        self.tokenized_corpus = None
    
    def index(self, corpus: Corpus) -> None:
        
        self.corpus = corpus
        
        if self.use_preprocessing and self.preprocessor:
            self.tokenized_corpus = [
                self.preprocessor.preprocess(doc.text)
                for doc in corpus
            ]
        else:
            self.tokenized_corpus = [
                doc.text.lower().split()
                for doc in corpus
            ]
        
        self.bm25 = BM25Okapi(
            self.tokenized_corpus,
            k1=self.k1,
            b=self.b
        )
        self._is_indexed = True
        
        print(f"[BM25] Indexed {len(corpus)} documents")
        print(f"[BM25] Average document length: {self.bm25.avgdl:.1f} tokens")
    
    def _retrieve_scores(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        
        if self.use_preprocessing and self.preprocessor:
            query_tokens = self.preprocessor.preprocess(query)
        else:
            query_tokens = query.lower().split()
        
        scores = self.bm25.get_scores(query_tokens)
        
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        return [(int(idx), float(scores[idx])) for idx in top_indices]
    
    def get_term_frequencies(self, doc_idx: int) -> dict:
        tokens = self.tokenized_corpus[doc_idx]
        freq = {}
        for token in tokens:
            freq[token] = freq.get(token, 0) + 1
        return freq
    
    def get_idf(self, term: str) -> float:
        if hasattr(self.bm25, 'idf'):
            return self.bm25.idf.get(term, 0.0)
        return 0.0
