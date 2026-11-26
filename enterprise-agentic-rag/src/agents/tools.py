from src.pipeline.rag_pipeline import RAGPipeline


class AgentTools:

    def __init__(
        self,
        rag_pipeline: RAGPipeline
    ):
        self.rag_pipeline = rag_pipeline

    def search_tool(
        self,
        query: str
    ):

        result = self.rag_pipeline.ask(query)

        return {
            "response": result["answer"]
        }

    def summary_tool(
        self,
        query: str
    ):

        prompt = f"""
Summarize the following topic.

Topic:
{query}

Provide:
1. Executive Summary
2. Key Points
3. Important Details
"""

        result = self.rag_pipeline.ask(prompt)

        return {
            "response": result["answer"]
        }

    def compare_tool(
        self,
        query: str
    ):

        prompt = f"""
Compare the entities mentioned below.

Question:
{query}

Provide:
1. Similarities
2. Differences
3. Recommendation

Use tabular format whenever possible.
"""

        result = self.rag_pipeline.ask(prompt)

        return {
            "response": result["answer"]
        }

    def multi_hop_tool(
        self,
        query: str
    ):

        prompt = f"""
Answer the question by combining findings
from multiple retrieved documents.

Question:
{query}

Provide a comprehensive answer.
"""

        result = self.rag_pipeline.ask(prompt)

        return {
            "response": result["answer"]
        }