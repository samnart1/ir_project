"""
Streamlit Web UI.

Run: streamlit run app.py

Features:
- Interactive search across research papers
- Side-by-side comparison of retrieval methods
- Real-time evaluation metrics
"""

import streamlit as st
import json
import time
from pathlib import Path

from src.corpus import Corpus
from src.retrievers import TFIDFRetriever, BM25Retriever
from src.evaluation.metrics import precision_at_k, recall_at_k, ndcg_at_k


@st.cache_resource
def load_corpus():
    return Corpus.load("data/corpus.json")


@st.cache_resource
def load_retrievers(_corpus):
    tfidf = TFIDFRetriever(max_features=5000, ngram_range=(1, 2))
    tfidf.index(_corpus)
    
    bm25 = BM25Retriever(k1=1.5, b=0.75)
    bm25.index(_corpus)
    
    return {"TF-IDF": tfidf, "BM25": bm25}


def main():
    st.set_page_config(
        page_title="Research Paper Search",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("Research Paper Information Retrieval")
    st.markdown("*Compare different retrieval methods on a corpus of ML papers*")
    
    corpus = load_corpus()
    retrievers = load_retrievers(corpus)
    
    st.sidebar.header("Settings")
    
    selected_retrievers = st.sidebar.multiselect(
        "Retrieval Methods",
        list(retrievers.keys()),
        default=list(retrievers.keys())
    )
    
    top_k = st.sidebar.slider("Number of Results", 3, 15, 5)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Corpus Statistics")
    st.sidebar.markdown(f"- **Documents:** {len(corpus)}")
    st.sidebar.markdown(f"- **Methods:** {len(retrievers)}")
    
    
    st.markdown("---")
    query = st.text_input("Enter your search query:", placeholder="e.g., transformer attention mechanism")
    
    if query:
        
        cols = st.columns(len(selected_retrievers))
        
        for idx, retriever_name in enumerate(selected_retrievers):
            retriever = retrievers[retriever_name]
            
            with cols[idx]:
                st.subheader(f"{retriever_name}")
                
                
                start_time = time.time()
                results = retriever.retrieve(query, top_k=top_k)
                elapsed_ms = (time.time() - start_time) * 1000
                
                st.caption(f"{elapsed_ms:.2f} ms")
                
                for result in results:
                    doc = result.document
                    with st.expander(f"**{result.rank}. {doc.title}** (score: {result.score:.4f})"):
                        st.markdown(f"**Authors:** {', '.join(doc.authors)}")
                        st.markdown(f"**Year:** {doc.year}")
                        st.markdown(f"**Categories:** {', '.join(doc.categories)}")
                        st.markdown("---")
                        st.markdown(doc.abstract)
        
        
        if len(selected_retrievers) > 1:
            st.markdown("---")
            st.subheader("Result Overlap Analysis")
            
            all_results = {}
            for name in selected_retrievers:
                results = retrievers[name].retrieve_ids(query, top_k)
                all_results[name] = set(results)
            
            
            names = list(all_results.keys())
            for i, name1 in enumerate(names):
                for name2 in names[i+1:]:
                    overlap = all_results[name1] & all_results[name2]
                    only_1 = all_results[name1] - all_results[name2]
                    only_2 = all_results[name2] - all_results[name1]
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(f"Common to both", len(overlap))
                    with col2:
                        st.metric(f"Only in {name1}", len(only_1))
                    with col3:
                        st.metric(f"Only in {name2}", len(only_2))
    
    else:
        
        st.markdown("### Try these sample queries:")
        sample_queries = [
            "transformer attention mechanism",
            "image classification neural networks",
            "word embeddings semantic representations",
            "reinforcement learning games",
            "recurrent neural networks LSTM"
        ]
        
        cols = st.columns(3)
        for i, sample in enumerate(sample_queries):
            with cols[i % 3]:
                if st.button(sample, key=f"sample_{i}"):
                    st.session_state['query'] = sample
                    st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>Information Retrieval | 
            Implements TF-IDF, BM25, Dense Retrieval & Hybrid methods</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
