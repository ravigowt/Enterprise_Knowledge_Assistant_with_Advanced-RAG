from typing import Dict, List

import json
from pathlib import Path

from langchain_core.documents import Document

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from src.memory.conversation_memory import ConversationMemory
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.llm_based_reranker import LLMReranker
from src.retrieval.hallucination_guard import HallucinationGuard
from src.utils.citation import CitationEngine

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.json"

def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

class RAGPipeline:

    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: LLMReranker,
        memory: ConversationMemory,
        hallucination_guard: HallucinationGuard,
        config_path=CONFIG_PATH
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.memory = memory
        self.hallucination_guard = hallucination_guard

        self.config = load_config(config_path)
        provider =  self.config.get("llm_provider", "openai").lower()
        temperature =  self.config.get("llm_temperature", 0.3)
    
        # --- ✅ Provider-specific model setup ---
        if provider == "openai":
            self.llm = ChatOpenAI(
                model=self.config.get("llm_rerank_model_name", "gpt-4o-mini"),
                temperature=temperature,
                streaming=True  # future-proof for streaming
            )
        elif provider == "gemini":
            self.llm = ChatGoogleGenerativeAI(
                model=self.config.get("llm_rerank_model_name", "gemini-2.5-flash"),
                temperature=temperature,
                streaming=True  # same reason as above
            )
        else:
            raise ValueError(f"❌ Unsupported LLM provider: {provider}")


    def _build_context(
        self,
        docs: List[Document]
    ) -> str:

        return "\n\n".join(
            [doc.page_content for doc in docs]
        )

    def _extract_sources(
        self,
        docs: List[Document]
    ) -> List[str]:

        sources = []

        for doc in docs:

            source = doc.metadata.get(
                "source",
                "Unknown"
            )

            if source not in sources:
                sources.append(source)

        return sources

    def _build_prompt(
        self,
        query: str,
        context: str,
        history: str
    ) -> str:

        return f"""
You are an Enterprise Knowledge Assistant.

Rules:
1. Answer ONLY from the provided context.
2. Use conversation history when relevant.
3. Do not make up information.
4. If the answer is unavailable, say:
   "I could not find that information in the available documents."

Conversation History:
{history}

Context:
{context}

Question:
{query}

Answer:
"""

    def ask(
        self,
        query: str,
        top_k: int = 3
    ) -> Dict:

        # =========================
        # Memory
        # =========================

        history = self.memory.get_context()

        # =========================
        # Retrieval
        # =========================

        retrieved_docs = (
            self.retriever.retrieve(
                query=query
            )
        )

        # =========================
        # Re-Ranking
        # =========================

        reranked_results = (
            self.reranker.rerank_with_scores(
                query=query,
                documents=retrieved_docs,
                top_k=top_k
            )
        )

        reranked_docs = [
            doc
            for doc, score
            in reranked_results
        ]

        # =========================
        # Context
        # =========================

        context = self._build_context(
            reranked_docs
        )

        prompt = self._build_prompt(
            query,
            context,
            history
        )


        # =========================
        # LLM Response
        # =========================
        response = self.llm.invoke(
            prompt
        )

        answer = response.content
        # =========================
        # Hallucination Guard
        # =========================

        guardrail_report = (
            self.hallucination_guard.evaluate(
                summary=answer,
                docs=reranked_docs
            )
        )
        
        if guardrail_report[0][3]:
            answer = (
                "I could not confidently verify "
                "this answer from the available documents."
            )

        # =========================
        # citations
        # =========================

        citation_engine = CitationEngine()
        citation_answer = citation_engine.generate_citations(answer,reranked_docs)

        
        
        # =========================
        # Save Memory
        # =========================

        self.memory.add_interaction(
            user_query=query,
            assistant_response=answer
        )
  
        return {
            "answer": citation_answer,
            "guardrail_passed":
                not(guardrail_report[0][3]),
            "documents_retrieved":
                len(retrieved_docs),
            "documents_reranked":
                len(reranked_docs)
        }
        