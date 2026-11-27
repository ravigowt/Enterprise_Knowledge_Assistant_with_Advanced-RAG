from typing import List, Dict


class ConversationMemory:

    def __init__(
        self,
        max_history: int = 10
    ):
        self.max_history = max_history
        self.history: List[Dict] = []

    def add_interaction(
        self,
        user_query: str,
        assistant_response: str
    ):

        self.history.append(
            {
                "user": user_query,
                "assistant": assistant_response
            }
        )

        if len(self.history) > self.max_history:
            self.history.pop(0)

    def get_history(self):

        return self.history

    def get_context(self):

        if not self.history:
            return ""

        conversation = []

        for item in self.history:

            conversation.append(
                f"User: {item['user']}"
            )

            conversation.append(
                f"Assistant: {item['assistant']}"
            )

        return "\n".join(conversation)

    def clear(self):
        self.history = []

    def last_n_interactions(
        self,
        n: int = 3
    ):
        return self.history[-n:]