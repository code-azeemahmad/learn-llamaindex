import os

import qdrant_client
from ingestion.pipeline import create_ingestion_pipeline, persist_pipeline_state
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    VectorStoreIndex,
)
from llama_index.core.query_engine import RetrieverQueryEngine, RouterQueryEngine
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.qdrant import QdrantVectorStore

# 1. SETUP GLOBAL MODELS
Settings.llm = Ollama(
    model="gemma4:26b",
    request_timeout=120.0,
)

Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text:latest",
    base_url="http://localhost:11434",
)

# Ensure sample python file exists for the second data source
os.makedirs("data", exist_ok=True)
if not os.path.exists("data/python.txt"):
    with open("data/python.txt", "w", encoding="utf-8") as f:
        f.write(
            "Python is a high-level, interpreted programming language known for readability.\n"
            "Common uses include Web Development (Django, FastAPI), Data Science (Pandas, NumPy),\n"
            "Machine Learning (PyTorch, TensorFlow), Automation, and Scripting.\n"
        )


# 2. BUILD ENGINE 1: ADVANCED ENTERPRISE RAG (Qdrant + BM25 + RRF + Reranker)
rag_documents = SimpleDirectoryReader(
    "data", exclude=["data/python.txt"]
).load_data()
for doc in rag_documents:
    if "file_name" in doc.metadata:
        doc.doc_id = doc.metadata["file_name"]

client = qdrant_client.QdrantClient(host="localhost", port=6333)
vector_store = QdrantVectorStore(client=client, collection_name="llamaindex_rag")

pipeline = create_ingestion_pipeline(vector_store=vector_store)
rag_nodes = pipeline.run(documents=rag_documents)
persist_pipeline_state(pipeline)

# Fallback to docstore values on cache hits
if not rag_nodes:
    rag_nodes = list(pipeline.docstore.docs.values())

rag_vector_index = VectorStoreIndex.from_vector_store(vector_store)
base_dense_retriever = rag_vector_index.as_retriever(similarity_top_k=5)

bm25_retriever = BM25Retriever.from_defaults(
    nodes=rag_nodes,
    similarity_top_k=5,
)

fusion_retriever = QueryFusionRetriever(
    retrievers=[base_dense_retriever, bm25_retriever],
    llm=Settings.llm,
    similarity_top_k=5,
    num_queries=3,
    mode=FUSION_MODES.RECIPROCAL_RANK,
    use_async=False,
    verbose=False,
)

reranker = SentenceTransformerRerank(
    model="BAAI/bge-reranker-base",
    top_n=3,
)

advanced_rag_engine = RetrieverQueryEngine.from_args(
    retriever=fusion_retriever,
    node_postprocessors=[reranker],
)

# Tool 1: Wrap Advanced RAG Engine
rag_tool = QueryEngineTool.from_defaults(
    query_engine=advanced_rag_engine,
    name="enterprise_rag_engine",
    description=(
        "Use this engine for questions about the indexed technical RAG knowledge base, "
        "system architecture, error codes (e.g., ERR-401), vector databases, and indexing pipelines."
    ),
)


# 3. BUILD ENGINE 2: PYTHON KNOWLEDGE BASE
python_documents = SimpleDirectoryReader(input_files=["data/python.txt"]).load_data()
python_index = VectorStoreIndex.from_documents(python_documents)
python_query_engine = python_index.as_query_engine()

# Tool 2: Wrap Python Query Engine
python_tool = QueryEngineTool.from_defaults(
    query_engine=python_query_engine,
    name="python_query_engine",
    description=(
        "Use this engine specifically for questions about Python programming, "
        "Python use cases, libraries, syntax, and Python software development."
    ),
)


# 4. CONSTRUCT ROUTER QUERY ENGINE
router_query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(llm=Settings.llm),
    query_engine_tools=[rag_tool, python_tool],
    # verbose=True,
)


# 5. PRACTICAL EVALUATION TASK
if __name__ == "__main__":
    queries = [
        "What is Retrieval-Augmented Generation?",
        "What are the main uses of Python?",
        "How does RRF combine multiple retrieval results?",
    ]

    print("\n==================== RUNNING ROUTER EVALUATION ====================\n")

    for idx, query in enumerate(queries, start=1):
        print(f"\n--- Query {idx}: '{query}' ---")
        response = router_query_engine.query(query)
        print("\n[Synthesized Answer]:")
        print(response)
        print("=" * 80)