from typing import List

from bm25_index import BM25Index
from llama_index.core import QueryBundle
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore


class BM25Retriever(BaseRetriever):

    def __init__(
        self,
        bm25_index: BM25Index,
        similarity_top_k: int = 5,
    ) -> None:
        self._bm25_index = bm25_index
        self._similarity_top_k = similarity_top_k
        super().__init__()

    def _retrieve(
        self,
        query_bundle: QueryBundle,
    ) -> List[NodeWithScore]:
        query = query_bundle.query_str

        results = self._bm25_index.search(
            query,
            top_k=self._similarity_top_k,
        )

        nodes = []
        for node_id, score in results:
            node = self._bm25_index.documents[node_id]
            nodes.append(
                NodeWithScore(
                    node=node,
                    score=score,
                )
            )

        return nodes

    async def _aretrieve(
        self,
        query_bundle: QueryBundle,
    ) -> List[NodeWithScore]:
        """Delegate async retrieval to the synchronous search method."""
        return self._retrieve(query_bundle)