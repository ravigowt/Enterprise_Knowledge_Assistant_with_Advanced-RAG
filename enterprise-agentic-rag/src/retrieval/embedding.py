import json
import os

from typing import Optional

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from pathlib import Path


# Load .env from the project root, no matter where the script is run from
load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")

# Load config.json from the project root (config/ folder)
CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.json"

def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_embedding_model(provider: str , embedded_model:str):

    provider = provider.lower().strip()

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env")

        print("Loading OpenAI embedding model...")

        return OpenAIEmbeddings(
            model=embedded_model,
            api_key=api_key
        )

    elif provider in ["gemini", "google"]:
        api_key = os.getenv("GOOGLE_API_KEY")

        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in .env")

        print("Loading Google embedding model...")

        return GoogleGenerativeAIEmbeddings(
            model=embedded_model,
            google_api_key=api_key
        )

    else:
        raise ValueError(
            f"Unsupported embedding provider: {provider}"
        )

class VectorStoreManager:

    def __init__(self, config: Optional[dict] = None):

        config = config or load_config()
        vector_cfg = config.get("vector_store", {})

        self.embeddings = get_embedding_model(
            config["embedding_provider"],
            config["embedding_model"]
        )

        self.db_type = vector_cfg["type"].lower()
        self.db_path = vector_cfg.get("local", {}).get("storage_path", "vectorstores/faiss_index")

    def create_index(self, documents):

        if self.db_type == "faiss":

            vector_store = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )

            vector_store.save_local(
                self.db_path
            )

            return vector_store

        elif self.db_type == "chroma":
            
            vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.db_path
            )

            return vector_store

        raise ValueError(
            f"Unsupported vector DB: {self.db_type}"
        )

    def load_index(self):

        if self.db_type == "faiss":

            return FAISS.load_local(
                self.db_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )

        elif self.db_type == "chroma":

            return Chroma(
                persist_directory=self.db_path,
                embedding_function=self.embeddings
            )

        raise ValueError(
            f"Unsupported vector DB: {self.db_type}"
        )

    def similarity_search(
        self,
        query: str,
        k: int = 5
    ):

        db = self.load_index()

        return db.similarity_search(
            query=query,
            k=k
        )

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5
    ):

        db = self.load_index()

        return db.similarity_search_with_score(
            query=query,
            k=k
        )

    def get_retriever(
        self,
        k: int = 5
    ):

        db = self.load_index()

        return db.as_retriever(
            search_kwargs={
                "k": k
            }
        )