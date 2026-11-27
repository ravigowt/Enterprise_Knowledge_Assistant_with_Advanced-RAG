import tempfile
from pathlib import Path

import streamlit as st

from src.document_loader import load_document
from src.chunking import split_documents

from src.retrieval.embedding import (
    VectorStoreManager
)

from src.retrieval.bm25_search import (
    BM25Retriever
)

from src.retrieval.hybrid_retriever import (
    HybridRetriever
)

from src.retrieval.llm_based_reranker import (
    LLMReranker
)

from src.retrieval.hallucination_guard import (
    HallucinationGuard
)

from src.memory.conversation_memory import (
    ConversationMemory
)

from src.pipeline.rag_pipeline import (
    RAGPipeline
)

from src.agents.graph import (
    EnterpriseAgent
)

# --------------------------------------------------
# Streamlit Config
# --------------------------------------------------

st.set_page_config(
    page_title="Enterprise Knowledge Assistant",
    layout="wide"
)

st.title("Enterprise Knowledge Assistant")

# --------------------------------------------------
# Session State
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

if "agent" not in st.session_state:
    st.session_state.agent = None

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.header("Knowledge Base")

    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )

    mode = st.radio(
        "Mode",
        [
            "Standard RAG",
            "Agentic AI"
        ]
    )

    process_btn = st.button(
        "Process & Index Documents"
    )

# --------------------------------------------------
# Build Knowledge Base
# --------------------------------------------------

if process_btn and uploaded_files:

    with st.spinner(
        "Processing documents..."
    ):

        docs = []
        chunks = []
        file_details = []
        with tempfile.TemporaryDirectory() as temp_dir:

            for file in uploaded_files:
                file_path = (
                    Path(temp_dir)
                    / file.name
                )

                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())

                file_details.append(file_path)

            for file in file_details:
                docs.append(load_document(
                    file
                ))

            for doc in docs:
                chunks.extend(split_documents(
                    doc
                ))

        # -------------------------
        # Vector Store
        # -------------------------

        vector_manager = (
            VectorStoreManager()
        )

        vector_manager.create_index(
            chunks
        )

        # -------------------------
        # BM25
        # -------------------------

        bm25 = BM25Retriever(
            chunks
        )

        # -------------------------
        # Hybrid Retriever
        # -------------------------

        hybrid = HybridRetriever(
            vector_manager,
            bm25
        )

        # -------------------------
        # Reranker
        # -------------------------

        reranker = LLMReranker()

        # -------------------------
        # Hallucination Guard
        # -------------------------

        guard = HallucinationGuard()

        # -------------------------
        # Memory
        # -------------------------

        memory = ConversationMemory()

        # -------------------------
        # RAG
        # -------------------------

        pipeline = RAGPipeline(
            retriever=hybrid,
            reranker=reranker,
            memory=memory,
            hallucination_guard=guard
        )

        # -------------------------
        # Agent
        # -------------------------

        agent = EnterpriseAgent(
            pipeline
        )

        st.session_state.pipeline = (
            pipeline
        )

        st.session_state.agent = (
            agent
        )

        st.success(
            f"""
Loaded {len(docs)} documents

Created {len(chunks)} chunks
"""
        )
        st.write("Start the Question...!")

# --------------------------------------------------
# Chat History
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):
        st.markdown(
            message["content"]
        )

# --------------------------------------------------
# Chat Input
# --------------------------------------------------

query = st.chat_input(
    "Ask a question..."
)

if query:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):

        if (
            st.session_state.pipeline
            is None
        ):

            st.warning(
                "Upload and process documents first."
            )

        else:

            with st.spinner(
                "Generating response..."
            ):

                if mode == "Standard RAG":

                    result = (
                        st.session_state.pipeline
                        .ask(query)
                    )

                else:

                    result = (
                        st.session_state.agent
                        .ask(query)
                    )

                answer = result.get(
                    "answer",
                    ""
                )

                st.markdown(answer)

                # Sources

                if result.get("sources"):

                    st.subheader(
                        "Sources"
                    )

                    for source in result[
                        "sources"
                    ]:

                        st.write(
                            f"📄 {source}"
                        )

                # Confidence

                if result.get(
                    "confidence"
                ) is not None:

                    st.metric(
                        label="Confidence",
                        value=f"{round(result['confidence'] * 100, 2)} %"
                    )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )