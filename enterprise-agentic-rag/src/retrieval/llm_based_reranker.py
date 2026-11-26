from typing import List, Tuple

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from pathlib import Path
import json

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.json"

def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

class LLMReranker:

    def __init__(
        self,
        config_path=CONFIG_PATH
        
    ):
        
        self.config = load_config(config_path)
        provider =  self.config.get("llm_provider", "openai").lower()
        temperature =  self.config.get("llm_temperature", 0.3)
    
        # --- ✅ Provider-specific model setup ---
        if provider == "openai":
            self.llm = ChatOpenAI(
                model=self.config.get("llm_model_name", "gpt-4o-mini"),
                temperature=temperature,
                streaming=True  # future-proof for streaming
            )
        elif provider == "gemini":
            self.llm = ChatGoogleGenerativeAI(
                model=self.config.get("llm_model_name", "gemini-2.5-flash"),
                temperature=temperature,
                streaming=True  # same reason as above
            )
        else:
            raise ValueError(f"❌ Unsupported LLM provider: {provider}")
        

    def _score_document(
        self,
        query: str,
        document: str
    ) -> float:

        prompt = f"""
You are a document relevance scorer.

Query:
{query}

Document:
{document}

Instructions:
- Rate relevance from 0 to 10.
- 10 = Perfectly answers the query.
- 0 = Completely unrelated.

Return ONLY a number.

Example:
8.5
"""

        response = self.llm.invoke(prompt)

        try:
            return float(response.content.strip())
        except Exception:
            return 0.0

    def rerank_with_scores(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 3
    ) -> List[Tuple[Document, float]]:

        scored_docs = []

        for doc in documents:

            score = self._score_document(
                query=query,
                document=doc.page_content
            )

            scored_docs.append(
                (doc, score)
            )

        scored_docs.sort(
            key=lambda x: x[1],
            reverse=True
        )

        return scored_docs[:top_k]

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 3
    ):

        ranked_docs = self.rerank_with_scores(
            query=query,
            documents=documents,
            top_k=top_k
        )

        return [
            doc
            for doc, score in ranked_docs
        ]