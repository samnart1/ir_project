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
    """
    Dense retriever using sentence transformer embeddings.
    
    Uses pre-trained language models to encode documents and queries
    into dense vector representations. Retrieval is performed using
    cosine similarity in the embedding space.
    
    Advantages over sparse methods:
    - Captures semantic similarity (synonyms, paraphrases)
    - No vocabulary mismatch problem
    - Works well with short queries
    
    Disadvantages:
    - Slower indexing (requires forward pass through neural network)
    - May miss exact keyword matches
    - Requires more memory for embeddings
    """
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 batch_size: int = 32,
                 normalize_embeddings: bool = True):
        """
        Initialize dense retriever.
        
        Args:
            model_name: Name of the sentence-transformer model
            batch_size: Batch size for encoding
            normalize_embeddings: Whether to L2-normalize embeddings
        """
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        
        self.model = None
        self.doc_embeddings = None
    
    def _load_model(self):
        """Lazy load the sentence transformer model."""
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
        """Build dense index from corpus."""
        self._load_model()
        self.corpus = corpus
        
        # Get document texts
        texts = corpus.get_texts()
        
        print(f"[Dense] Encoding {len(texts)} documents...")
        
        # Encode documents
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
        """Retrieve documents using embedding similarity."""
        # Encode query
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=self.normalize_embeddings
        )[0]
        
        # Calculate cosine similarity
        if self.normalize_embeddings:
            # For normalized vectors, dot product = cosine similarity
            similarities = np.dot(self.doc_embeddings, query_embedding)
        else:
            # Calculate cosine similarity manually
            query_norm = np.linalg.norm(query_embedding)
            doc_norms = np.linalg.norm(self.doc_embeddings, axis=1)
            similarities = np.dot(self.doc_embeddings, query_embedding) / (doc_norms * query_norm)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return [(int(idx), float(similarities[idx])) for idx in top_indices]
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get the embedding for a text."""
        self._load_model()
        return self.model.encode([text], normalize_embeddings=self.normalize_embeddings)[0]
    
    def get_document_embedding(self, doc_idx: int) -> np.ndarray:
        """Get the embedding for a document by index."""
        return self.doc_embeddings[doc_idx]
    
    def similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts."""
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)
        return float(np.dot(emb1, emb2))
