from src.agents.router import QueryRouter


router = QueryRouter()


def router_node(state):

    route = router.classify(
        state["question"]
    )

    return {
        **state,
        "route": route,
        "query": state["question"]
    }


def search_node(state):

    result = state["tools"].search_tool(
        state["query"]
    )

    return {
        **state,
        "answer": result["response"],
    }


def summary_node(state):

    result = state["tools"].summary_tool(
        state["query"]
    )

    return {
        **state,
        "answer": result["response"],
    }


def compare_node(state):

    result = state["tools"].compare_tool(
        state["query"]
    )

    return {
        **state,
        "answer": result["response"],
    }


def multihop_node(state):

    result = state["tools"].multi_hop_tool(
        state["query"]
    )

    return {
        **state,
        "answer": result["response"],
    }