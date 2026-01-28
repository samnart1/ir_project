"""
TF-IDF (Term Frequency-Inverse Document Frequency) Retriever.

Implements the classic TF-IDF weighting scheme for document retrieval
using scikit-learn's TfidfVectorizer.
"""

from typing import List, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .base import BaseRetriever
from ..corpus import Corpus, TextPreprocessor


class TFIDFRetriever(BaseRetriever):
    """
    TF-IDF based retriever using cosine similarity.
    
    TF-IDF weighs terms by:
    - TF (Term Frequency): How often a term appears in a document
    - IDF (Inverse Document Frequency): How rare a term is across all documents
    
    This helps identify discriminative terms that are frequent in a document
    but rare overall.
    """
    
    def __init__(self, 
                 max_features: int = 10000,
                 ngram_range: Tuple[int, int] = (1, 2),
                 use_preprocessing: bool = True):
        """
        Initialize TF-IDF retriever.
        
        Args:
            max_features: Maximum vocabulary size
            ngram_range: Range of n-grams to consider (min, max)
            use_preprocessing: Whether to apply custom preprocessing
        """
        super().__init__()
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.use_preprocessing = use_preprocessing
        
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            lowercase=True
        )
        self.preprocessor = TextPreprocessor() if use_preprocessing else None
        self.doc_vectors = None
    
    def index(self, corpus: Corpus) -> None:
        """Build TF-IDF index from corpus."""
        self.corpus = corpus
        
        # Get document texts
        if self.use_preprocessing and self.preprocessor:
            texts = [self.preprocessor.preprocess_to_string(doc.text) 
                    for doc in corpus]
        else:
            texts = corpus.get_texts()
        
        # Build TF-IDF matrix
        self.doc_vectors = self.vectorizer.fit_transform(texts)
        self._is_indexed = True
        
        print(f"[TF-IDF] Indexed {len(corpus)} documents")
        print(f"[TF-IDF] Vocabulary size: {len(self.vectorizer.vocabulary_)}")
    
    def _retrieve_scores(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """Retrieve documents using cosine similarity."""
        # Preprocess query
        if self.use_preprocessing and self.preprocessor:
            query = self.preprocessor.preprocess_to_string(query)
        
        # Transform query to TF-IDF vector
        query_vector = self.vectorizer.transform([query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, self.doc_vectors).flatten()
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return [(int(idx), float(similarities[idx])) for idx in top_indices]
    
    def get_feature_names(self) -> List[str]:
        """Get the vocabulary terms."""
        return self.vectorizer.get_feature_names_out().tolist()
    
    def get_document_vector(self, doc_idx: int) -> np.ndarray:
        """Get the TF-IDF vector for a document."""
        return self.doc_vectors[doc_idx].toarray().flatten()
