#!/usr/bin/env python3
"""
ArXiv Dataset Loader

Downloads and processes the ArXiv Paper Abstracts dataset from Kaggle.
Creates a subset of 500-1000 papers for the IR project.

Usage:
    1. Download dataset from: https://www.kaggle.com/datasets/spsayakpaul/arxiv-paper-abstracts
    2. Place the CSV file in data/raw/
    3. Run: python load_arxiv_data.py

Alternatively, if you have Kaggle API configured:
    python load_arxiv_data.py --download
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import List, Dict, Optional
import subprocess


def download_from_kaggle(dataset_name: str = "spsayakpaul/arxiv-paper-abstracts", 
                         output_dir: str = "data/raw"):
    """Download dataset using Kaggle API."""
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        print(f"Downloading {dataset_name} from Kaggle...")
        subprocess.run([
            "kaggle", "datasets", "download", "-d", dataset_name,
            "-p", output_dir, "--unzip"
        ], check=True)
        print("Download complete!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error downloading: {e}")
        print("\nManual download instructions:")
        print(f"1. Go to: https://www.kaggle.com/datasets/{dataset_name}")
        print("2. Download the dataset")
        print(f"3. Extract to {output_dir}/")
        return False
    except FileNotFoundError:
        print("Kaggle CLI not found. Install with: pip install kaggle")
        print("\nManual download instructions:")
        print(f"1. Go to: https://www.kaggle.com/datasets/{dataset_name}")
        print("2. Download the dataset")
        print(f"3. Extract to {output_dir}/")
        return False


def load_arxiv_csv(filepath: str) -> List[Dict]:
    """Load ArXiv data from CSV file."""
    import csv
    
    papers = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            papers.append(row)
    
    return papers


def load_arxiv_json(filepath: str) -> List[Dict]:
    """Load ArXiv data from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def process_arxiv_data(raw_data: List[Dict], 
                       num_papers: int = 1000,
                       categories: Optional[List[str]] = None,
                       seed: int = 42) -> List[Dict]:
    """
    Process and filter ArXiv data.
    
    Args:
        raw_data: Raw paper data from Kaggle
        num_papers: Number of papers to include (500-1000 recommended)
        categories: Filter to specific categories (e.g., ['cs.CL', 'cs.LG', 'cs.CV'])
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    processed = []
    
    for i, paper in enumerate(raw_data):
        # Handle different field names from different dataset versions
        doc_id = paper.get('id') or paper.get('paper_id') or f"arxiv_{i}"
        title = paper.get('title') or paper.get('titles') or ""
        abstract = paper.get('abstract') or paper.get('summaries') or paper.get('abstracts') or ""
        
        # Get categories/terms
        cats = paper.get('categories') or paper.get('terms') or paper.get('category') or ""
        if isinstance(cats, str):
            cats = [c.strip() for c in cats.replace("'", "").strip("[]").split(',')]
        
        # Get authors
        authors = paper.get('authors') or paper.get('author') or ""
        if isinstance(authors, str):
            authors = [a.strip() for a in authors.replace("'", "").strip("[]").split(',')]
        
        # Get year if available
        year = paper.get('year') or paper.get('update_date', '')[:4] if paper.get('update_date') else None
        try:
            year = int(year) if year else None
        except:
            year = None
        
        # Skip if missing essential fields
        if not title or not abstract:
            continue
        
        # Clean up text
        title = title.strip().replace('\n', ' ')
        abstract = abstract.strip().replace('\n', ' ')
        
        # Filter by category if specified
        if categories:
            if not any(cat in cats for cat in categories):
                continue
        
        processed.append({
            "doc_id": str(doc_id),
            "title": title,
            "abstract": abstract,
            "authors": authors[:5] if isinstance(authors, list) else [authors],  # Limit authors
            "categories": cats[:3] if isinstance(cats, list) else [cats],  # Limit categories
            "year": year
        })
    
    # Sample if we have more than needed
    if len(processed) > num_papers:
        processed = random.sample(processed, num_papers)
    
    print(f"Processed {len(processed)} papers")
    return processed


def generate_queries_from_corpus(corpus: List[Dict], num_queries: int = 25) -> List[Dict]:
    """
    Generate evaluation queries based on the corpus.
    
    Creates queries from paper titles and samples relevant documents.
    You should manually review and adjust these!
    """
    random.seed(42)
    
    queries = []
    
    # Sample papers to create queries from
    sample_papers = random.sample(corpus, min(num_queries * 2, len(corpus)))
    
    # Common IR query patterns
    query_templates = [
        lambda p: extract_key_terms(p['title'], 3),
        lambda p: extract_key_terms(p['abstract'], 4),
        lambda p: f"{extract_key_terms(p['title'], 2)} {p['categories'][0] if p['categories'] else ''}".strip(),
    ]
    
    for i, paper in enumerate(sample_papers[:num_queries]):
        template = query_templates[i % len(query_templates)]
        query_text = template(paper)
        
        if not query_text or len(query_text) < 5:
            continue
        
        # The source paper is definitely relevant
        relevant = [paper['doc_id']]
        
        # Find similar papers (simple keyword overlap)
        query_words = set(query_text.lower().split())
        for other in corpus:
            if other['doc_id'] == paper['doc_id']:
                continue
            other_words = set(other['title'].lower().split() + other['abstract'].lower().split()[:50])
            overlap = len(query_words & other_words)
            if overlap >= 2 and len(relevant) < 5:
                relevant.append(other['doc_id'])
        
        queries.append({
            "query_id": f"q{i+1:02d}",
            "query_text": query_text,
            "relevant_docs": relevant,
            "graded_relevance": {relevant[0]: 3, **{d: 2 for d in relevant[1:]}}
        })
    
    return queries


def extract_key_terms(text: str, num_terms: int = 3) -> str:
    """Extract key terms from text (simple stopword removal)."""
    stopwords = {
        'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'it',
        'its', 'we', 'our', 'their', 'which', 'what', 'who', 'whom', 'how', 'when',
        'where', 'why', 'all', 'each', 'every', 'both', 'few', 'more', 'most',
        'other', 'some', 'such', 'no', 'not', 'only', 'same', 'so', 'than', 'too',
        'very', 'just', 'also', 'now', 'new', 'using', 'based', 'via', 'into'
    }
    
    words = text.lower().split()
    # Remove punctuation and filter
    words = [w.strip('.,!?()[]{}":;') for w in words]
    words = [w for w in words if w and len(w) > 2 and w not in stopwords]
    
    return ' '.join(words[:num_terms])


def main():
    parser = argparse.ArgumentParser(description="Load and process ArXiv dataset")
    parser.add_argument('--download', action='store_true', help='Download from Kaggle')
    parser.add_argument('--input', type=str, help='Input file path (CSV or JSON)')
    parser.add_argument('--num-papers', type=int, default=1000, help='Number of papers (default: 1000)')
    parser.add_argument('--num-queries', type=int, default=25, help='Number of test queries (default: 25)')
    parser.add_argument('--categories', type=str, nargs='+', 
                        default=['cs.CL', 'cs.LG', 'cs.CV', 'cs.AI', 'cs.NE', 'stat.ML'],
                        help='Filter to categories (default: ML/AI related)')
    parser.add_argument('--output-dir', type=str, default='data', help='Output directory')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(f"{args.output_dir}/raw", exist_ok=True)
    
    # Download if requested
    if args.download:
        download_from_kaggle(output_dir=f"{args.output_dir}/raw")
    
    # Find input file
    input_file = args.input
    if not input_file:
        # Look for common file names
        raw_dir = Path(f"{args.output_dir}/raw")
        for pattern in ['*.csv', '*.json', 'arxiv*.csv', 'arxiv*.json']:
            files = list(raw_dir.glob(pattern))
            if files:
                input_file = str(files[0])
                break
    
    if not input_file or not os.path.exists(input_file):
        print("=" * 60)
        print("INPUT FILE NOT FOUND")
        print("=" * 60)
        print("\nPlease download the ArXiv dataset:")
        print("1. Go to: https://www.kaggle.com/datasets/spsayakpaul/arxiv-paper-abstracts")
        print("2. Download and extract the CSV file")
        print(f"3. Place it in: {args.output_dir}/raw/")
        print("\nOr run with --download flag if you have Kaggle API configured")
        print("\nAlternatively, specify input file: python load_arxiv_data.py --input path/to/file.csv")
        return
    
    print(f"Loading data from: {input_file}")
    
    # Load raw data
    if input_file.endswith('.csv'):
        raw_data = load_arxiv_csv(input_file)
    else:
        raw_data = load_arxiv_json(input_file)
    
    print(f"Loaded {len(raw_data)} raw papers")
    
    # Process data
    corpus = process_arxiv_data(
        raw_data,
        num_papers=args.num_papers,
        categories=args.categories
    )
    
    if not corpus:
        print("ERROR: No papers matched the criteria!")
        print("Try removing the --categories filter or check your input file format")
        return
    
    # Save corpus
    corpus_path = f"{args.output_dir}/corpus.json"
    with open(corpus_path, 'w', encoding='utf-8') as f:
        json.dump(corpus, f, indent=2, ensure_ascii=False)
    print(f"Saved corpus to: {corpus_path}")
    
    # Generate queries
    queries = generate_queries_from_corpus(corpus, num_queries=args.num_queries)
    
    # Save queries
    queries_path = f"{args.output_dir}/queries.json"
    with open(queries_path, 'w', encoding='utf-8') as f:
        json.dump(queries, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(queries)} queries to: {queries_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("DATASET READY")
    print("=" * 60)
    print(f"Corpus: {len(corpus)} papers")
    print(f"Queries: {len(queries)} test queries")
    print(f"\nCategories distribution:")
    
    cat_counts = {}
    for paper in corpus:
        for cat in paper.get('categories', []):
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
    
    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {cat}: {count}")
    
    print(f"\nYou can now run: python main.py --mode evaluate")
    print("\nNOTE: Review data/queries.json and adjust relevance judgments manually")
    print("      for more accurate evaluation!")


if __name__ == "__main__":
    main()
