"""
Unit tests for the Information Retrieval system.

Run with: pytest tests/test_ir_system.py -v
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.corpus import Document, Corpus, TextPreprocessor
from src.evaluation.metrics import (
    precision_at_k, recall_at_k, f1_at_k,
    average_precision, ndcg_at_k, reciprocal_rank
)


class TestDocument:
    def test_document_creation(self):
        doc = Document(
            doc_id="test1",
            title="Test Title",
            abstract="Test abstract content"
        )
        assert doc.doc_id == "test1"
        assert doc.text == "Test Title Test abstract content"
    
    def test_document_serialization(self):
        doc = Document(
            doc_id="test1",
            title="Test",
            abstract="Abstract",
            authors=["Author1"],
            year=2024
        )
        data = doc.to_dict()
        doc2 = Document.from_dict(data)
        assert doc.doc_id == doc2.doc_id
        assert doc.title == doc2.title


class TestCorpus:
    def test_corpus_operations(self):
        docs = [
            Document(doc_id="1", title="First", abstract="Content one"),
            Document(doc_id="2", title="Second", abstract="Content two"),
        ]
        corpus = Corpus(docs)
        
        assert len(corpus) == 2
        assert corpus.get_by_id("1").title == "First"
        assert corpus.get_by_index(1).title == "Second"


class TestPreprocessor:
    def test_preprocessing(self):
        preprocessor = TextPreprocessor()
        tokens = preprocessor.preprocess("The quick brown fox jumps!")
        
        assert "the" not in tokens  # stopword removed
        assert len(tokens) > 0


class TestMetrics:
    def test_precision_at_k(self):
        retrieved = ["d1", "d2", "d3", "d4", "d5"]
        relevant = {"d1", "d3", "d5"}
        
        assert precision_at_k(retrieved, relevant, 1) == 1.0
        assert precision_at_k(retrieved, relevant, 2) == 0.5
        assert precision_at_k(retrieved, relevant, 5) == 0.6
    
    def test_recall_at_k(self):
        retrieved = ["d1", "d2", "d3", "d4", "d5"]
        relevant = {"d1", "d3", "d5"}
        
        assert recall_at_k(retrieved, relevant, 1) == pytest.approx(1/3)
        assert recall_at_k(retrieved, relevant, 5) == 1.0
    
    def test_f1_at_k(self):
        retrieved = ["d1", "d2", "d3"]
        relevant = {"d1", "d3"}
        
        p = precision_at_k(retrieved, relevant, 3)
        r = recall_at_k(retrieved, relevant, 3)
        expected_f1 = 2 * p * r / (p + r)
        
        assert f1_at_k(retrieved, relevant, 3) == pytest.approx(expected_f1)
    
    def test_average_precision(self):
        # Perfect ranking
        retrieved = ["d1", "d2", "d3"]
        relevant = {"d1", "d2", "d3"}
        assert average_precision(retrieved, relevant) == 1.0
        
        # Imperfect ranking
        retrieved = ["d1", "x", "d2", "x", "d3"]
        relevant = {"d1", "d2", "d3"}
        ap = average_precision(retrieved, relevant)
        assert 0 < ap < 1
    
    def test_reciprocal_rank(self):
        retrieved = ["x", "x", "d1", "x"]
        relevant = {"d1"}
        assert reciprocal_rank(retrieved, relevant) == pytest.approx(1/3)
        
        # First position
        retrieved = ["d1", "x", "x"]
        assert reciprocal_rank(retrieved, relevant) == 1.0
    
    def test_ndcg(self):
        retrieved = ["d1", "d2", "d3"]
        relevance_scores = {"d1": 3, "d2": 2, "d3": 1}
        
        # Perfect ranking should give NDCG = 1
        ndcg = ndcg_at_k(retrieved, relevance_scores, 3)
        assert ndcg == pytest.approx(1.0)
        
        # Reversed ranking should give lower NDCG
        retrieved_bad = ["d3", "d2", "d1"]
        ndcg_bad = ndcg_at_k(retrieved_bad, relevance_scores, 3)
        assert ndcg_bad < ndcg


class TestIntegration:
    @pytest.fixture
    def sample_corpus(self):
        docs = [
            Document("1", "Neural Networks Introduction", "Deep learning fundamentals and neural network architectures"),
            Document("2", "Natural Language Processing", "Text processing and language understanding with transformers"),
            Document("3", "Computer Vision Basics", "Image recognition and convolutional neural networks"),
        ]
        return Corpus(docs)
    
    def test_tfidf_retriever(self, sample_corpus):
        from src.retrievers import TFIDFRetriever
        
        retriever = TFIDFRetriever()
        retriever.index(sample_corpus)
        
        results = retriever.retrieve("neural networks deep learning", top_k=2)
        assert len(results) == 2
        assert results[0].rank == 1
        assert results[0].document.doc_id in ["1", "2", "3"]
    
    def test_bm25_retriever(self, sample_corpus):
        from src.retrievers import BM25Retriever
        
        retriever = BM25Retriever()
        retriever.index(sample_corpus)
        
        results = retriever.retrieve("language processing", top_k=2)
        assert len(results) == 2
        # NLP doc should rank high
        assert any(r.document.doc_id == "2" for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
