from langgraph.graph import (
    StateGraph,
    END
)

from src.agents.state import AgentState

from src.agents.tools import AgentTools

from src.agents.nodes import (
    router_node,
    search_node,
    summary_node,
    compare_node,
    multihop_node
)


class EnterpriseAgent:

    def __init__(
        self,
        rag_pipeline
    ):

        self.tools = AgentTools(
            rag_pipeline
        )

        workflow = StateGraph(
            AgentState
        )

        workflow.add_node(
            "router",
            router_node
        )

        workflow.add_node(
            "simple",
            search_node
        )

        workflow.add_node(
            "summary",
            summary_node
        )

        workflow.add_node(
            "compare",
            compare_node
        )

        workflow.add_node(
            "multihop",
            multihop_node
        )

        workflow.set_entry_point(
            "router"
        )

        workflow.add_conditional_edges(
            "router",
            self.route_decision,
            {
                "simple": "simple",
                "summary": "summary",
                "compare": "compare",
                "multihop": "multihop"
            }
        )

        workflow.add_edge(
            "simple",
            END
        )

        workflow.add_edge(
            "summary",
            END
        )

        workflow.add_edge(
            "compare",
            END
        )

        workflow.add_edge(
            "multihop",
            END
        )

        self.graph = workflow.compile()

    def route_decision(
        self,
        state
    ):

        return state["route"]

    def ask(
        self,
        question: str
    ):

        result = self.graph.invoke(
            {
                "question": question,
                "tools": self.tools
            }
        )

        return {
            "answer": result["answer"]
        }