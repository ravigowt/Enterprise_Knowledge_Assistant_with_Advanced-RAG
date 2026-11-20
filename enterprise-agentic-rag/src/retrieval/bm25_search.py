from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from pathlib import Path
import json

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.json"

def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

class BM25Retriever:
    def __init__(self, documents: list[Document], config_path=CONFIG_PATH):
        """
        Initialize BM25 with document collection.
        """
        self.config = load_config(config_path)
        self.documents = documents

        self.tokenized_corpus = [
            doc.page_content.lower().split()
            for doc in documents
        ]

        self.bm25 = BM25Okapi(
            self.tokenized_corpus,
            k1=self.config["bm25"]["k1"],
            b=self.config["bm25"]["b"],
        )

    def search(
        self,
        query: str,
        k: int = 5
    ):
        """
        Returns top-k documents with BM25 scores.
        """

        tokenized_query = query.lower().split()

        scores = self.bm25.get_scores(
            tokenized_query
        )

        ranked_docs = sorted(
            zip(self.documents, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return ranked_docs[:k]

    def search_documents(
        self,
        query: str,
        k: int = 5
    ):
        """
        Returns only documents.
        """

        results = self.search(
            query=query,
            k=k
        )

        return [
            doc
            for doc, score in results
        ]