# learn-llamaindex\main.py
import qdrant_client
from bm25_index import BM25Index
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
)
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.postprocessor.sbert_rerank import (
    SentenceTransformerRerank,
)
from llama_index.vector_stores.qdrant import QdrantVectorStore
from retrievers.bm25_retriever import BM25Retriever

llm = Settings.llm = Ollama(
    model="gemma4:26b",
    request_timeout=120.0,
)

Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text:latest",
    base_url="http://localhost:11434",
)

documents = SimpleDirectoryReader("data").load_data()
print(f"Loaded documents: {len(documents)}")

pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(
            chunk_size=256,
            chunk_overlap=20
        ),
    ]
)

nodes = pipeline.run(documents=documents)
print(f"Number of nodes: {len(nodes)}")


bm25_index = BM25Index()

bm25_index.add_documents(nodes)

bm25_retriever = BM25Retriever(
    bm25_index=bm25_index,
    similarity_top_k=5,
)


# bm25_nodes = bm25_retriever.retrieve(
#     "What is Retrieval-Augmented Generation?"
# )

# print("\n========== BM25 RETRIEVER ==========")

# for i, node_with_score in enumerate(bm25_nodes):

#     print(f"\n--- Node {i} ---")
#     print("Score:", node_with_score.score)
#     print("Text:")
#     print(node_with_score.node.text)


client = qdrant_client.QdrantClient(
    host="localhost",
    port=6333
)

vector_store = QdrantVectorStore(
    client=client,
    collection_name="llamaindex_rag",
)

storage_context = StorageContext.from_defaults(
    vector_store=vector_store,
)

similarity_filter = SimilarityPostprocessor(
    similarity_cutoff=0.30,
)

# Reranker after retrieval
reranker = SentenceTransformerRerank(
    model="BAAI/bge-reranker-base",
    top_n=3,
)

# Metadata Filtering
filters = MetadataFilters(
    filters=[
        MetadataFilter(
            key="department",
            value="engineering",
        ),
        MetadataFilter(
            key="document_type",
            value="policy",
        )
    ]
)

# Build index
index = VectorStoreIndex(
    nodes,
    storage_context=storage_context
)

base_retriever = index.as_retriever(
    similarity_top_k=5,
)


# QueryFusionRetriever Abstraction
fusion_retriever = QueryFusionRetriever(
    retrievers=[
        base_retriever,
        bm25_retriever,
    ],
    llm=Settings.llm,
    similarity_top_k=5,
    num_queries=4,
    mode=FUSION_MODES.RECIPROCAL_RANK,
    use_async=False,
    verbose=True,
)

query = "How can RAG improves Enterprise AI applications?"


query_engine = RetrieverQueryEngine.from_args(
    retriever=fusion_retriever,
    node_postprocessors=[
        similarity_filter,
        reranker,
    ],
)


response = query_engine.query(
    query
)

print("\nAnswer:")
print(response)

# for source in response.source_nodes:
#     print( 
#         source.score,
#         source.node.text,
#     )
