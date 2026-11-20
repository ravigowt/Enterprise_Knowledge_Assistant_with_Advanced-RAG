from typing import List, Tuple

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from pathlib import Path
import json

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.json"

def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class Reranker:

    def __init__(
        self,
        config_path=CONFIG_PATH
    ):
        self.config = load_config(config_path)
        self.model_name = self.config["cross_encoder_model_name"]
        print(f"Loading reranker model : {self.model_name}")
        self.model = CrossEncoder(self.model_name)

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 3
    ) -> List[Document]:

        if not documents:
            return []

        pairs = [
            (query, doc.page_content)
            for doc in documents
        ]

        scores = self.model.predict(pairs)

        ranked_docs = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [
            doc
            for doc, score in ranked_docs[:top_k]
        ]

    def rerank_with_scores(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 3
    ) -> List[Tuple[Document, float]]:

        if not documents:
            return []

        pairs = [
            (query, doc.page_content)
            for doc in documents
        ]

        scores = self.model.predict(pairs)

        ranked_docs = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return ranked_docs[:top_k]