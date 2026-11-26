from pathlib import Path
import json

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.json"

def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_llm():
    config = load_config(CONFIG_PATH)
    provider =  config.get("llm_provider", "openai").lower()
    temperature =  config.get("llm_temperature", 0.3)

    # --- ✅ Provider-specific model setup ---
    if provider == "openai":
        return ChatOpenAI(
            model=config.get("llm_model_name", "gpt-4o-mini"),
            temperature=temperature,
            streaming=True  # future-proof for streaming
        )
    elif provider == "gemini":
        return ChatGoogleGenerativeAI(
            model=config.get("llm_model_name", "gemini-2.5-flash"),
            temperature=temperature,
            streaming=True  # same reason as above
        )
    else:
        raise ValueError(f"❌ Unsupported LLM provider: {provider}")


class QueryRouter:

    def __init__(self):
        self.llm = get_llm()

    def classify(
        self,
        question: str
    ) -> str:

        prompt = f"""
Classify the query.

Return ONLY one word:

simple
summary
compare
multihop

Examples:

What is VPN policy?
-> simple

Summarize leave policy
-> summary

Compare annual leave and sick leave
-> compare

Can I access VPN from a personal laptop
while working from home?
-> multihop

Question:
{question}
"""

        response = self.llm.invoke(prompt)

        intent = response.content.strip().lower()

        allowed = [
            "simple",
            "summary",
            "compare",
            "multihop"
        ]

        if intent not in allowed:
            intent = "simple"

        return intent