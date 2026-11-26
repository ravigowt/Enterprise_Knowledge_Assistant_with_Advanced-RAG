import re
import numpy as np
from typing import List, Tuple, Dict
from langchain_core.documents import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class CitationEngine:
    """
    Matches factual sentences to their most relevant source chunks
    and appends inline citations like [policy_updates.txt].
    """

    def __init__(self, top_k: int = 2):
        self.top_k = top_k
        self.vectorizer = TfidfVectorizer(stop_words="english")

    # -----------------------------------------------------
    # Sentence splitter
    # -----------------------------------------------------
    def _split_sentences(self, text: str) -> List[str]:
        # Splits the input text into sentences using punctuation (., !, ?) as boundaries,
        # then returns only those sentences that have more than 3 words.
        sents = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s for s in sents if len(s.split()) > 3]

    # -----------------------------------------------------
    # Build TF-IDF embedding for source docs
    # -----------------------------------------------------
    def _build_source_matrix(self, docs: List[Document]):
        # Converts the content of each document into a list (corpus),
        # then creates a TF-IDF matrix (X) representing the importance of each word in each document.
        # Returns both the matrix and the original text list.
        corpus = [d.page_content for d in docs]
        X = self.vectorizer.fit_transform(corpus)
        return X, corpus

    # -----------------------------------------------------
    # Match sentences → source docs
    # -----------------------------------------------------
    def generate_citations(self, summary: str, docs: List[Document]) -> str:
        sentences = self._split_sentences(summary)
        X, corpus = self._build_source_matrix(docs)

        citations = []

        for sent in sentences:

            # Converts each summary sentence into a TF-IDF vector, computes its similarity to each source document,
            # and finds the indices of the top-k most similar documents.
            
            q_vec = self.vectorizer.transform([sent])
            sims = cosine_similarity(q_vec, X).flatten()
            top_indices = np.argsort(sims)[::-1][:self.top_k]

            # For each of the top similar documents (with similarity above 0.05),
            # get the source filename from its metadata and add it to the list of matched sources.
            matched_sources = [
                docs[i].metadata.get("source", "unknown").split("/")[-1]
                for i in top_indices
                if sims[i] > 0.05
            ]

            # Unique, readable
            matched_sources = list(dict.fromkeys(matched_sources))
            # Suppose matched_sources is: ["file1.txt", "file2.txt", "file1.txt", "file3.txt"]
            # After this line runs: matched_sources = ["file1.txt", "file2.txt", "file3.txt"]

            # Append inline citation
            citation_str = ", ".join(matched_sources[:self.top_k])
            if citation_str:
                citations.append(f"{sent} [{citation_str}]")
            else:
                citations.append(sent)

        return " ".join(citations)
