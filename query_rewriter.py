from llama_index.llms.ollama import Ollama


class QueryRewriter:
    def __init__(self, llm: Ollama):
        self.llm = llm

    def rewrite(self, query: str, num_queries: int = 3) -> list[str]:
        prompt = f"""
        Generate {num_queries} different search queries for the user's question according to the terminologies mentioned in the query.

        User query:
        {query}

        Return only the queries, one per line. No numbering, no extra text.
        """
        response = self.llm.complete(prompt)

        queries = [
            line.strip() for line in str(response).strip().split("\n") if line.strip()
        ]
        return queries