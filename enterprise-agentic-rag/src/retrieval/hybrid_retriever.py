from typing import List
from langchain_core.documents import Document


class HybridRetriever:
    def __init__(
        self,
        vector_manager,
        bm25_retriever
    ):
        self.vector_manager = vector_manager
        self.bm25_retriever = bm25_retriever

    def retrieve(
        self,
        query: str,
        vector_k: int = 5,
        bm25_k: int = 5
    ) -> List[Document]:

        # Semantic Search
        vector_docs = self.vector_manager.similarity_search(
            query=query,
            k=vector_k
        )

        # BM25 Search
        bm25_results = self.bm25_retriever.search(
            query=query,
            k=bm25_k
        )

        bm25_docs = [
            doc for doc, score in bm25_results
        ]

        # Merge Results
        combined = vector_docs + bm25_docs

        # Remove Duplicates
        unique_docs = []
        seen = set()

        for doc in combined:

            key = (
                doc.page_content,
                str(doc.metadata)
            )

            if key not in seen:
                seen.add(key)
                unique_docs.append(doc)

        return unique_docs