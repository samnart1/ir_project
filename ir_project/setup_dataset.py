#!/usr/bin/env python3
"""
Load ArXiv dataset from Hugging Face (no authentication required).

This is the easiest way to get started - just run:
    python setup_dataset.py

It will download ~1000 ML/AI papers and create the evaluation queries.
"""

import json
import os
import random
from typing import List, Dict, Optional


def load_from_huggingface(num_papers: int = 1000, 
                          categories: List[str] = None,
                          seed: int = 42) -> List[Dict]:
    """
    Load ArXiv papers from Hugging Face datasets.
    
    Args:
        num_papers: Target number of papers
        categories: Filter to specific categories
        seed: Random seed
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Installing datasets library...")
        import subprocess
        subprocess.run(["pip", "install", "datasets", "--break-system-packages", "-q"])
        from datasets import load_dataset
    
    print("Loading ArXiv dataset from Hugging Face...")
    print("(This may take a few minutes on first run)")
    
    # Load the dataset (streaming to avoid downloading entire 1.7M papers)
    dataset = load_dataset(
        "arxiv-community/arxiv_dataset",
        split="train",
        streaming=True,
        trust_remote_code=True
    )
    
    # Default to ML/AI categories
    if categories is None:
        categories = ['cs.CL', 'cs.LG', 'cs.CV', 'cs.AI', 'cs.NE', 'stat.ML', 
                      'cs.IR', 'cs.HC', 'cs.RO']
    
    papers = []
    seen_ids = set()
    
    print(f"Filtering for categories: {categories}")
    print(f"Target: {num_papers} papers")
    
    for i, paper in enumerate(dataset):
        if len(papers) >= num_papers * 3:  # Get extra to sample from
            break
        
        if i % 10000 == 0 and i > 0:
            print(f"  Scanned {i} papers, found {len(papers)} matching...")
        
        # Get categories
        paper_cats = paper.get('categories', '')
        if isinstance(paper_cats, str):
            paper_cats = paper_cats.split()
        
        # Check if any category matches
        if not any(cat in paper_cats for cat in categories):
            continue
        
        # Skip duplicates
        paper_id = paper.get('id', '')
        if paper_id in seen_ids:
            continue
        seen_ids.add(paper_id)
        
        # Get abstract
        abstract = paper.get('abstract', '')
        title = paper.get('title', '')
        
        # Skip if missing essential fields
        if not abstract or not title or len(abstract) < 100:
            continue
        
        # Clean text
        title = ' '.join(title.split())
        abstract = ' '.join(abstract.split())
        
        # Parse authors
        authors_raw = paper.get('authors_parsed', []) or paper.get('authors', '')
        if isinstance(authors_raw, list) and authors_raw:
            if isinstance(authors_raw[0], list):
                authors = [' '.join(a[:2]).strip() for a in authors_raw[:5]]
            else:
                authors = authors_raw[:5]
        elif isinstance(authors_raw, str):
            authors = [a.strip() for a in authors_raw.split(',')[:5]]
        else:
            authors = []
        
        papers.append({
            "doc_id": paper_id,
            "title": title,
            "abstract": abstract[:2000],  # Limit abstract length
            "authors": authors,
            "categories": paper_cats[:3] if isinstance(paper_cats, list) else paper_cats.split()[:3],
            "year": int(paper.get('update_date', '2020')[:4]) if paper.get('update_date') else None
        })
    
    # Sample to target size
    random.seed(seed)
    if len(papers) > num_papers:
        papers = random.sample(papers, num_papers)
    
    print(f"Final corpus: {len(papers)} papers")
    return papers


def generate_evaluation_queries(corpus: List[Dict], num_queries: int = 30) -> List[Dict]:
    """
    Generate evaluation queries with relevance judgments.
    
    Strategy:
    1. Extract key phrases from titles
    2. Find similar papers based on term overlap
    3. Create graded relevance (3=exact match, 2=similar, 1=related)
    """
    random.seed(42)
    
    stopwords = {
        'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these',
        'those', 'it', 'its', 'we', 'our', 'their', 'which', 'what', 'who',
        'using', 'based', 'via', 'paper', 'propose', 'proposed', 'show',
        'approach', 'method', 'methods', 'results', 'model', 'models', 'new'
    }
    
    def get_keywords(text: str, n: int = 5) -> List[str]:
        words = text.lower().split()
        words = [w.strip('.,!?()[]{}":;') for w in words]
        words = [w for w in words if w and len(w) > 2 and w not in stopwords]
        return words[:n]
    
    def compute_similarity(kw1: List[str], text: str) -> int:
        text_words = set(text.lower().split())
        return sum(1 for kw in kw1 if kw in text_words)
    
    # Build keyword index for all papers
    paper_keywords = {}
    for paper in corpus:
        combined = f"{paper['title']} {paper['abstract']}"
        paper_keywords[paper['doc_id']] = set(get_keywords(combined, 20))
    
    # Select diverse seed papers for queries
    selected = random.sample(corpus, min(num_queries + 10, len(corpus)))
    
    queries = []
    used_keywords = set()
    
    for paper in selected:
        if len(queries) >= num_queries:
            break
        
        # Generate query from title keywords
        title_kw = get_keywords(paper['title'], 4)
        query_text = ' '.join(title_kw)
        
        # Skip if too similar to existing queries
        if query_text in used_keywords or len(query_text) < 8:
            continue
        used_keywords.add(query_text)
        
        # Find relevant documents
        query_kw_set = set(title_kw)
        relevant = []
        
        for other in corpus:
            if len(relevant) >= 8:
                break
            
            overlap = len(query_kw_set & paper_keywords[other['doc_id']])
            
            if other['doc_id'] == paper['doc_id']:
                relevant.append((other['doc_id'], 3))  # Source paper = highly relevant
            elif overlap >= 3:
                relevant.append((other['doc_id'], 2))  # Good overlap
            elif overlap >= 2:
                relevant.append((other['doc_id'], 1))  # Some overlap
        
        if len(relevant) >= 2:
            queries.append({
                "query_id": f"q{len(queries)+1:02d}",
                "query_text": query_text,
                "relevant_docs": [doc_id for doc_id, _ in relevant],
                "graded_relevance": {doc_id: score for doc_id, score in relevant}
            })
    
    return queries


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup ArXiv dataset for IR project")
    parser.add_argument('--num-papers', type=int, default=1000, 
                        help='Number of papers (default: 1000)')
    parser.add_argument('--num-queries', type=int, default=30,
                        help='Number of evaluation queries (default: 30)')
    parser.add_argument('--output-dir', type=str, default='data',
                        help='Output directory (default: data)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load papers
    print("=" * 60)
    print("SETTING UP ARXIV DATASET")
    print("=" * 60)
    
    corpus = load_from_huggingface(num_papers=args.num_papers)
    
    if not corpus:
        print("ERROR: Failed to load corpus!")
        return
    
    # Save corpus
    corpus_path = os.path.join(args.output_dir, 'corpus.json')
    with open(corpus_path, 'w', encoding='utf-8') as f:
        json.dump(corpus, f, indent=2, ensure_ascii=False)
    print(f"\nSaved corpus to: {corpus_path}")
    
    # Generate queries
    print("\nGenerating evaluation queries...")
    queries = generate_evaluation_queries(corpus, num_queries=args.num_queries)
    
    # Save queries
    queries_path = os.path.join(args.output_dir, 'queries.json')
    with open(queries_path, 'w', encoding='utf-8') as f:
        json.dump(queries, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(queries)} queries to: {queries_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SETUP COMPLETE")
    print("=" * 60)
    print(f"Corpus size: {len(corpus)} papers")
    print(f"Test queries: {len(queries)}")
    
    # Category distribution
    cat_counts = {}
    for paper in corpus:
        for cat in paper.get('categories', []):
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
    
    print(f"\nTop categories:")
    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1])[:8]:
        print(f"  {cat}: {count}")
    
    # Year distribution
    years = [p['year'] for p in corpus if p.get('year')]
    if years:
        print(f"\nYear range: {min(years)} - {max(years)}")
    
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("1. Run evaluation:  python main.py --mode evaluate")
    print("2. With dense:      python main.py --mode evaluate --dense")
    print("3. Interactive:     python main.py --mode search")
    print("4. Web UI:          streamlit run app.py")
    print("\nNOTE: Review data/queries.json to verify/adjust relevance judgments")


if __name__ == "__main__":
    main()
