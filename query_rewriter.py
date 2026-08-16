from llama_index.llms.ollama import Ollama


class QueryRewriter:
    def __init__(self, llm: Ollama):
        self.llm = llm

    def rewrite(
        self,
        query: str,
        conversation: str,
    ) -> str:

        prompt = f"""
Rewrite the user's query into a standalone
search query.

Conversation:
{conversation}

User query:
{query}

Return only the rewritten search query.
"""

        response = self.llm.complete(prompt)

        return str(response).strip()