# Research Paper Information Retrieval System

A comprehensive Information Retrieval system implementing multiple retrieval methods with proper evaluation using standard IR metrics.

## 📋 Project Overview

This project implements and compares **four retrieval methods** on a corpus of **1000 ML/AI research papers**:

| Method | Type | Description |
|--------|------|-------------|
| **TF-IDF** | Sparse | Classic term weighting with cosine similarity |
| **BM25** | Sparse | Probabilistic model with length normalization |
| **Dense** | Neural | Semantic search using sentence embeddings |
| **Hybrid** | Combined | Reciprocal Rank Fusion of BM25 + Dense |

All methods are evaluated using **standard IR metrics**:
- Precision@k, Recall@k, F1@k
- MAP (Mean Average Precision)
- NDCG (Normalized Discounted Cumulative Gain)
- MRR (Mean Reciprocal Rank)

## 📊 Dataset

- **Corpus**: 1000 ML/AI research papers (titles + abstracts)
- **Categories**: cs.CL, cs.CV, cs.LG, cs.AI, cs.IR
- **Evaluation queries**: 30 test queries with relevance judgments

To use a different dataset from Kaggle:
```bash
# Option 1: If you have Kaggle API configured
python load_arxiv_data.py --download --num-papers 1000

# Option 2: Manual download
# 1. Download from https://www.kaggle.com/datasets/spsayakpaul/arxiv-paper-abstracts
# 2. Place CSV in data/raw/
# 3. Run: python load_arxiv_data.py --input data/raw/arxiv.csv
```

## 🏗️ Project Structure

```
ir_project/
├── main.py                    # CLI interface
├── app.py                     # Streamlit web UI
├── requirements.txt           # Dependencies
├── data/
│   ├── corpus.json           # Document corpus (25 ML papers)
│   └── queries.json          # Evaluation queries with relevance judgments
├── src/
│   ├── __init__.py
│   ├── corpus.py             # Document & corpus handling
│   ├── retrievers/
│   │   ├── __init__.py
│   │   ├── base.py           # Base retriever interface
│   │   ├── tfidf.py          # TF-IDF implementation
│   │   ├── bm25.py           # BM25 implementation
│   │   ├── dense.py          # Dense retrieval (sentence-transformers)
│   │   └── hybrid.py         # Hybrid RRF fusion
│   └── evaluation/
│       ├── __init__.py
│       ├── metrics.py        # IR evaluation metrics
│       └── benchmark.py      # Benchmarking suite
└── tests/                    # Unit tests
```

## 🚀 Quick Start

### Installation

```bash
# Clone/download the project
cd ir_project

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (automatic on first run)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### Usage

```bash
# Quick demo (fast, sparse methods only)
python main.py --mode demo

# Full evaluation (sparse methods)
python main.py --mode evaluate

# Full evaluation including neural methods (slower but comprehensive)
python main.py --mode evaluate --dense

# Interactive search
python main.py --mode search

# Web interface
streamlit run app.py
```

## 📊 Retrieval Methods

### 1. TF-IDF (Term Frequency-Inverse Document Frequency)

Classic vector space model that weighs terms by:
- **TF**: How often a term appears in a document
- **IDF**: How rare a term is across the corpus

```python
from src.retrievers import TFIDFRetriever

retriever = TFIDFRetriever(max_features=10000, ngram_range=(1, 2))
retriever.index(corpus)
results = retriever.retrieve("transformer attention", top_k=10)
```

### 2. BM25 (Best Matching 25)

Probabilistic model improving on TF-IDF with:
- **Term frequency saturation**: Diminishing returns for repeated terms
- **Length normalization**: Accounts for document length bias

```python
from src.retrievers import BM25Retriever

retriever = BM25Retriever(k1=1.5, b=0.75)
retriever.index(corpus)
results = retriever.retrieve("neural network training", top_k=10)
```

### 3. Dense Retrieval (Sentence Transformers)

Neural embedding-based retrieval using pre-trained transformers:
- Captures **semantic similarity** (synonyms, paraphrases)
- No vocabulary mismatch problem
- Uses `all-MiniLM-L6-v2` model by default

```python
from src.retrievers import DenseRetriever

retriever = DenseRetriever(model_name="all-MiniLM-L6-v2")
retriever.index(corpus)
results = retriever.retrieve("deep learning image recognition", top_k=10)
```

### 4. Hybrid Retrieval (RRF Fusion)

Combines sparse and dense methods using **Reciprocal Rank Fusion**:
- Leverages exact matching from BM25
- Leverages semantic understanding from Dense
- Often outperforms either method alone

```python
from src.retrievers import HybridRetriever

retriever = HybridRetriever(fusion_method="rrf")
retriever.index(corpus)
results = retriever.retrieve("language model pretraining", top_k=10)
```

## 📈 Evaluation Metrics

### Implemented Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| **Precision@k** | `\|retrieved ∩ relevant\| / k` | Fraction of top-k that are relevant |
| **Recall@k** | `\|retrieved ∩ relevant\| / \|relevant\|` | Fraction of relevant docs in top-k |
| **F1@k** | `2 × P × R / (P + R)` | Harmonic mean of P and R |
| **AP** | `Σ P@k × rel(k) / \|relevant\|` | Average precision at each relevant doc |
| **MAP** | `mean(AP)` over all queries | Mean Average Precision |
| **NDCG@k** | `DCG@k / IDCG@k` | Handles graded relevance |
| **MRR** | `mean(1/rank of first relevant)` | Mean Reciprocal Rank |

### Running Evaluation

```python
from src.evaluation import Benchmark
from src.retrievers import BM25Retriever, TFIDFRetriever

# Load benchmark
benchmark = Benchmark.load("data/queries.json")

# Evaluate single retriever
results = benchmark.evaluate(bm25_retriever)

# Compare multiple retrievers
comparison = benchmark.compare([tfidf, bm25, dense, hybrid])
```

## 📝 Sample Output

```
======================================================================
COMPARISON SUMMARY
======================================================================
Retriever                  MAP        MRR       P@5      P@10      R@10   NDCG@10   Time(ms)
----------------------------------------------------------------------
TFIDFRetriever          0.7234     0.8667    0.5333    0.3000    0.8444    0.7891       1.23
BM25Retriever           0.7856     0.9333    0.5867    0.3133    0.8889    0.8234       0.89
DenseRetriever          0.8123     0.9000    0.6000    0.3200    0.9111    0.8456      12.34
HybridRetriever         0.8567     0.9333    0.6267    0.3333    0.9333    0.8789      15.67
======================================================================
```

## 🎯 Key Features

1. **Modular Design**: Easy to add new retrievers or metrics
2. **Proper Evaluation**: Uses standard IR benchmarking methodology
3. **Multiple Fusion Methods**: RRF and weighted score combination
4. **Interactive UI**: Both CLI and web interface
5. **Graded Relevance**: Supports both binary and graded relevance judgments

## 📚 Extending the System

### Adding a New Retriever

```python
from src.retrievers.base import BaseRetriever

class MyRetriever(BaseRetriever):
    def index(self, corpus):
        # Build your index
        self._is_indexed = True
    
    def _retrieve_scores(self, query, top_k):
        # Return list of (doc_index, score) tuples
        return [(0, 1.0), (1, 0.8), ...]
```

### Adding Custom Corpus

```python
from src.corpus import Document, Corpus

docs = [
    Document(doc_id="1", title="...", abstract="..."),
    Document(doc_id="2", title="...", abstract="..."),
]
corpus = Corpus(docs)
corpus.save("data/my_corpus.json")
```

## 🔬 Theoretical Background

### Why Multiple Methods?

| Method | Strengths | Weaknesses |
|--------|-----------|------------|
| TF-IDF | Fast, interpretable | No length normalization |
| BM25 | Better ranking, handles length | Still keyword-based |
| Dense | Semantic understanding | Slower, needs GPU for large scale |
| Hybrid | Best of both worlds | More complex |

### Reciprocal Rank Fusion (RRF)

RRF combines rankings without requiring score normalization:

```
RRF_score(d) = Σ 1 / (k + rank(d))
```

where `k` is typically 60. This method is robust and often outperforms
individual methods.

## 📖 References

1. Robertson, S. E., & Walker, S. (1994). Some simple effective approximations to the 2-Poisson model for probabilistic weighted retrieval. *SIGIR*.
2. Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. *EMNLP*.
3. Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009). Reciprocal rank fusion outperforms condorcet and individual rank learning methods. *SIGIR*.

## 📄 License

MIT License - Free to use for academic purposes.

---

*Built for Information Retrieval Course*
