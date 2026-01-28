"""
Corpus handling and preprocessing for the IR system.

This module provides the Document and Corpus classes for loading,
preprocessing, and managing research paper collections.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional

# Fallback stopwords if NLTK is unavailable
ENGLISH_STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he',
    'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's",
    'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
    'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
    'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
    'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
    'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
    'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will',
    'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're',
    've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't",
    'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't",
    'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn',
    "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren',
    "weren't", 'won', "won't", 'wouldn', "wouldn't"
}

# Try to use NLTK, fall back to simple tokenization
try:
    import nltk
    from nltk.corpus import stopwords as nltk_stopwords
    from nltk.stem import PorterStemmer
    from nltk.tokenize import word_tokenize
    
    # Try to load NLTK data
    try:
        nltk_stopwords.words('english')
        NLTK_AVAILABLE = True
    except:
        NLTK_AVAILABLE = False
except ImportError:
    NLTK_AVAILABLE = False


@dataclass
class Document:
    """Represents a single document in the corpus."""
    doc_id: str
    title: str
    abstract: str
    authors: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    year: Optional[int] = None
    
    @property
    def text(self) -> str:
        """Concatenated title and abstract for retrieval."""
        return f"{self.title} {self.abstract}"
    
    def to_dict(self) -> Dict:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "abstract": self.abstract,
            "authors": self.authors,
            "categories": self.categories,
            "year": self.year
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Document":
        return cls(
            doc_id=data["doc_id"],
            title=data["title"],
            abstract=data["abstract"],
            authors=data.get("authors", []),
            categories=data.get("categories", []),
            year=data.get("year")
        )


class TextPreprocessor:
    """Handles text preprocessing for IR."""
    
    def __init__(self, 
                 lowercase: bool = True,
                 remove_punctuation: bool = True,
                 remove_stopwords: bool = True,
                 stem: bool = True):
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.stem = stem
        
        # Set up stemmer
        if stem and NLTK_AVAILABLE:
            self.stemmer = PorterStemmer()
        else:
            self.stemmer = None
        
        # Set up stopwords
        if remove_stopwords:
            if NLTK_AVAILABLE:
                try:
                    self.stop_words = set(nltk_stopwords.words('english'))
                except:
                    self.stop_words = ENGLISH_STOPWORDS
            else:
                self.stop_words = ENGLISH_STOPWORDS
        else:
            self.stop_words = set()
    
    def _simple_tokenize(self, text: str) -> List[str]:
        """Simple whitespace and punctuation-based tokenization."""
        # Split on whitespace and punctuation
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def _simple_stem(self, word: str) -> str:
        """Very simple suffix stripping."""
        # Basic suffix removal
        suffixes = ['ing', 'ed', 'ly', 'es', 's', 'ment', 'ness', 'tion', 'ation']
        for suffix in suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                return word[:-len(suffix)]
        return word
    
    def preprocess(self, text: str) -> List[str]:
        """Preprocess text and return list of tokens."""
        if self.lowercase:
            text = text.lower()
        
        if self.remove_punctuation:
            text = re.sub(r'[^\w\s]', ' ', text)
        
        # Tokenize
        if NLTK_AVAILABLE:
            try:
                tokens = word_tokenize(text)
            except:
                tokens = self._simple_tokenize(text)
        else:
            tokens = self._simple_tokenize(text)
        
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stop_words]
        
        if self.stem:
            if self.stemmer:
                tokens = [self.stemmer.stem(t) for t in tokens]
            else:
                tokens = [self._simple_stem(t) for t in tokens]
        
        # Remove empty tokens and single characters
        tokens = [t for t in tokens if len(t) > 1]
        
        return tokens
    
    def preprocess_to_string(self, text: str) -> str:
        """Preprocess and return as space-joined string."""
        return ' '.join(self.preprocess(text))


class Corpus:
    """Manages a collection of documents."""
    
    def __init__(self, documents: Optional[List[Document]] = None):
        self.documents: List[Document] = documents or []
        self._id_to_idx: Dict[str, int] = {}
        self._rebuild_index()
    
    def _rebuild_index(self):
        """Rebuild the document ID to index mapping."""
        self._id_to_idx = {doc.doc_id: idx for idx, doc in enumerate(self.documents)}
    
    def add_document(self, doc: Document):
        """Add a document to the corpus."""
        self._id_to_idx[doc.doc_id] = len(self.documents)
        self.documents.append(doc)
    
    def get_by_id(self, doc_id: str) -> Optional[Document]:
        """Retrieve a document by its ID."""
        idx = self._id_to_idx.get(doc_id)
        return self.documents[idx] if idx is not None else None
    
    def get_by_index(self, idx: int) -> Document:
        """Retrieve a document by its index."""
        return self.documents[idx]
    
    def get_texts(self) -> List[str]:
        """Get all document texts."""
        return [doc.text for doc in self.documents]
    
    def get_ids(self) -> List[str]:
        """Get all document IDs."""
        return [doc.doc_id for doc in self.documents]
    
    def __len__(self) -> int:
        return len(self.documents)
    
    def __iter__(self):
        return iter(self.documents)
    
    def save(self, path: str):
        """Save corpus to JSON file."""
        with open(path, 'w') as f:
            json.dump([doc.to_dict() for doc in self.documents], f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "Corpus":
        """Load corpus from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        documents = [Document.from_dict(d) for d in data]
        return cls(documents)
    
    @classmethod
    def from_json(cls, data: List[Dict]) -> "Corpus":
        """Create corpus from list of dictionaries."""
        documents = [Document.from_dict(d) for d in data]
        return cls(documents)
