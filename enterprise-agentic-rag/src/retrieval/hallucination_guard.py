# utils/hallucination_guard.py

import re
import numpy as np
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.documents import Document


class HallucinationGuard:
    """
    Detects unsupported or low-confidence claims in a summary
    by comparing sentences to retrieved context.
    """

    def __init__(self, threshold: float = 0.25):
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.threshold = threshold

    # ----------------------------
    # Sentence splitting
    # ----------------------------
    def _split_sentences(self, text: str) -> List[str]:
        sents = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s for s in sents if len(s.split()) > 3]

    # ----------------------------
    # Compute semantic support
    # ----------------------------
    def evaluate(self, summary: str, docs: List[Document]) -> List[Tuple[str, float, str]]:
        """
        Compare each summary sentence against retrieved docs
        and return (sentence, max_score, status).
        """
        sentences = self._split_sentences(summary)
        corpus = [d.page_content for d in docs]

        if not corpus or not sentences:
            return []

        X = self.vectorizer.fit_transform(corpus)

        results = []
        for sent in sentences:
            q_vec = self.vectorizer.transform([sent])
            sims = cosine_similarity(q_vec, X).flatten()
            max_sim = float(np.max(sims)) if len(sims) > 0 else 0.0

            status = "✅ Supported" if max_sim >= self.threshold else "⚠️ Possibly Hallucinated"
            hallucination_result = (max_sim < self.threshold)
            results.append((sent, round(max_sim, 3), status, hallucination_result))

        return results