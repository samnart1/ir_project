#!/usr/bin/env python3
"""
Research Paper Information Retrieval System

A comprehensive IR system implementing multiple retrieval methods:
- TF-IDF (Term Frequency-Inverse Document Frequency)
- BM25 (Best Matching 25)
- Dense Retrieval (Sentence Transformers)
- Hybrid (Reciprocal Rank Fusion)

With proper evaluation using standard IR metrics:
- Precision@k, Recall@k, F1@k
- MAP (Mean Average Precision)
- NDCG (Normalized Discounted Cumulative Gain)
- MRR (Mean Reciprocal Rank)

Usage:
    python main.py --mode evaluate    # Run full evaluation
    python main.py --mode search      # Interactive search
    python main.py --mode demo        # Quick demo
"""

import argparse
import json
from pathlib import Path

from src.corpus import Corpus
from src.retrievers import TFIDFRetriever, BM25Retriever, DenseRetriever, HybridRetriever
from src.evaluation import Benchmark, QueryRelevance


def load_corpus(path: str = "data/corpus.json") -> Corpus:
    """Load the document corpus."""
    print(f"Loading corpus from {path}...")
    return Corpus.load(path)


def load_benchmark(path: str = "data/queries.json") -> Benchmark:
    """Load the evaluation benchmark."""
    print(f"Loading queries from {path}...")
    with open(path, 'r') as f:
        data = json.load(f)
    
    queries = [QueryRelevance.from_dict(q) for q in data]
    return Benchmark(queries=queries, k_values=[1, 3, 5, 10])


def run_evaluation(use_dense: bool = True):
    """
    Run full evaluation comparing all retrieval methods.
    
    This is the main evaluation that compares:
    - TF-IDF baseline
    - BM25 (improved sparse method)
    - Dense retrieval (semantic, uses neural embeddings)
    - Hybrid (combines BM25 + Dense using RRF)
    """
    print("=" * 70)
    print("INFORMATION RETRIEVAL SYSTEM EVALUATION")
    print("=" * 70)
    
    # Load data
    corpus = load_corpus()
    benchmark = load_benchmark()
    
    print(f"\nCorpus size: {len(corpus)} documents")
    print(f"Number of queries: {len(benchmark.queries)}")
    
    # Initialize retrievers
    print("\n" + "=" * 70)
    print("INDEXING PHASE")
    print("=" * 70)
    
    retrievers = []
    
    # TF-IDF
    print("\n[1/4] Building TF-IDF index...")
    tfidf = TFIDFRetriever(max_features=5000, ngram_range=(1, 2))
    tfidf.index(corpus)
    retrievers.append(tfidf)
    
    # BM25
    print("\n[2/4] Building BM25 index...")
    bm25 = BM25Retriever(k1=1.5, b=0.75)
    bm25.index(corpus)
    retrievers.append(bm25)
    
    if use_dense:
        # Dense
        print("\n[3/4] Building Dense index (this may take a moment)...")
        dense = DenseRetriever(model_name="all-MiniLM-L6-v2")
        dense.index(corpus)
        retrievers.append(dense)
        
        # Hybrid
        print("\n[4/4] Building Hybrid index...")
        hybrid = HybridRetriever(
            sparse_retriever=BM25Retriever(k1=1.5, b=0.75),
            dense_retriever=DenseRetriever(model_name="all-MiniLM-L6-v2"),
            fusion_method="rrf"
        )
        hybrid.index(corpus)
        retrievers.append(hybrid)
    else:
        print("\n[3/4] Skipping Dense retriever (use --dense to enable)")
        print("[4/4] Skipping Hybrid retriever")
    
    # Run comparison
    print("\n" + "=" * 70)
    print("EVALUATION PHASE")
    print("=" * 70)
    
    results = benchmark.compare(retrievers)
    
    # Save results
    results_path = "evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump({name: {k: float(v) for k, v in metrics.items()} 
                   for name, metrics in results.items()}, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    return results


def run_interactive_search(use_dense: bool = False):
    """Run interactive search mode."""
    print("=" * 70)
    print("INTERACTIVE SEARCH MODE")
    print("=" * 70)
    
    corpus = load_corpus()
    
    # Use BM25 for fast interactive search
    print("\nBuilding search index...")
    retriever = BM25Retriever()
    retriever.index(corpus)
    
    print("\nReady! Enter your search queries (type 'quit' to exit)")
    print("-" * 70)
    
    while True:
        query = input("\nQuery: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not query:
            continue
        
        results = retriever.retrieve(query, top_k=5)
        
        print(f"\nTop 5 results for: '{query}'")
        print("-" * 50)
        
        for result in results:
            doc = result.document
            print(f"\n[{result.rank}] {doc.title} (score: {result.score:.4f})")
            print(f"    Authors: {', '.join(doc.authors[:2])}{'...' if len(doc.authors) > 2 else ''}")
            print(f"    Year: {doc.year}")
            # Truncate abstract
            abstract = doc.abstract[:150] + "..." if len(doc.abstract) > 150 else doc.abstract
            print(f"    {abstract}")


def run_demo():
    """Run a quick demo showing the system capabilities."""
    print("=" * 70)
    print("INFORMATION RETRIEVAL DEMO")
    print("=" * 70)
    
    corpus = load_corpus()
    
    # Build indices
    print("\nBuilding search indices...")
    
    bm25 = BM25Retriever()
    bm25.index(corpus)
    
    tfidf = TFIDFRetriever()
    tfidf.index(corpus)
    
    # Demo queries
    demo_queries = [
        "transformer attention mechanism",
        "image classification deep learning",
        "word embeddings semantic"
    ]
    
    print("\n" + "=" * 70)
    print("DEMO SEARCH RESULTS")
    print("=" * 70)
    
    for query in demo_queries:
        print(f"\n{'='*60}")
        print(f"Query: '{query}'")
        print("=" * 60)
        
        print("\nBM25 Results:")
        for result in bm25.retrieve(query, top_k=3):
            print(f"  [{result.rank}] {result.document.title} (score: {result.score:.4f})")
        
        print("\nTF-IDF Results:")
        for result in tfidf.retrieve(query, top_k=3):
            print(f"  [{result.rank}] {result.document.title} (score: {result.score:.4f})")
    
    print("\n" + "=" * 70)
    print("Demo complete! Run with --mode evaluate for full evaluation.")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Research Paper Information Retrieval System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode demo              Quick demo
  python main.py --mode evaluate          Full evaluation (sparse only)
  python main.py --mode evaluate --dense  Full evaluation (includes neural methods)
  python main.py --mode search            Interactive search
        """
    )
    
    parser.add_argument(
        '--mode', 
        choices=['evaluate', 'search', 'demo'],
        default='demo',
        help='Operation mode (default: demo)'
    )
    
    parser.add_argument(
        '--dense',
        action='store_true',
        help='Include dense/neural retrievers (slower but more comprehensive)'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'evaluate':
        run_evaluation(use_dense=args.dense)
    elif args.mode == 'search':
        run_interactive_search(use_dense=args.dense)
    else:
        run_demo()


if __name__ == "__main__":
    main()
