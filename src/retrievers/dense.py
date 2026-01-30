"""
Dense Retriever using Sentence Transformers.

Implements neural embedding-based retrieval using pre-trained
transformer models. Documents and queries are encoded into dense
vectors, and retrieval is done via similarity search.
"""

from typing import List, Tuple, Optional
import numpy as np

from .base import BaseRetriever
from ..corpus import Corpus


class DenseRetriever(BaseRetriever):
    
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 batch_size: int = 32,
                 normalize_embeddings: bool = True):
        
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        
        self.model = None
        self.doc_embeddings = None
    
    def _load_model(self):
        
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
                print(f"[Dense] Loading model: {self.model_name}")
                self.model = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for DenseRetriever. "
                    "Install with: pip install sentence-transformers"
                )
    
    def index(self, corpus: Corpus) -> None:
        
        self._load_model()
        self.corpus = corpus
        
        
        texts = corpus.get_texts()
        
        print(f"[Dense] Encoding {len(texts)} documents...")
        
        self.doc_embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=self.normalize_embeddings
        )
        
        self._is_indexed = True
        print(f"[Dense] Indexed {len(corpus)} documents")
        print(f"[Dense] Embedding dimension: {self.doc_embeddings.shape[1]}")
    
    def _retrieve_scores(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=self.normalize_embeddings
        )[0]
        
        
        if self.normalize_embeddings:
            
            similarities = np.dot(self.doc_embeddings, query_embedding)
        else:
            
            query_norm = np.linalg.norm(query_embedding)
            doc_norms = np.linalg.norm(self.doc_embeddings, axis=1)
            similarities = np.dot(self.doc_embeddings, query_embedding) / (doc_norms * query_norm)
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return [(int(idx), float(similarities[idx])) for idx in top_indices]
    
    def get_embedding(self, text: str) -> np.ndarray:
        
        self._load_model()
        return self.model.encode([text], normalize_embeddings=self.normalize_embeddings)[0]
    
    def get_document_embedding(self, doc_idx: int) -> np.ndarray:
        
        return self.doc_embeddings[doc_idx]
    
    def similarity(self, text1: str, text2: str) -> float:
        
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)
        return float(np.dot(emb1, emb2))
