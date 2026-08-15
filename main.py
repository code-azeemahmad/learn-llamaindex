import qdrant_client
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.qdrant import QdrantVectorStore

# LLM
Settings.llm = Ollama(
    model="gemma4:26b",
    request_timeout=120.0,
)

# Embedding model
Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text:latest",
    base_url="http://localhost:11434",
)

# Load documents
documents = SimpleDirectoryReader("data").load_data()
print(f"Loaded documents: {len(documents)}")

# Ingestion Pipeline
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

# Build index
index = VectorStoreIndex(
    nodes, 
    storage_context=storage_context
)

retriever = index.as_retriever(
    similarity_top_k=5
)

# nodes = retriever.retrieve(
#     "What is Retrieval-Augmented Generation?"
# )

# print("\nBefore filtering:")

# for item in nodes:
#     print(
#         f"Score: {item.score:.4f}"
#     )

# postprocessor = SimilarityPostprocessor(
#     similarity_cutoff=0.60,
# )

# filtered_nodes = postprocessor.postprocess_nodes(
#     nodes
# )

# print("\nAfter filtering:")

# for item in filtered_nodes:
#     print(
#         f"Score: {item.score:.4f}"
#     )

similarity_filter = SimilarityPostprocessor(
    similarity_cutoff=0.750,
)

query_engine = index.as_query_engine(
    similarity_top_k=3,
    SimilarityPostprocessor=[
        similarity_filter,
    ]
)

# Ask question
response = query_engine.query(
    "What role does a vector database play in RAG?"
)

print("\nAnswer:")
print(response)

for source in response.source_nodes:
    print(
        "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++   ",
        source.score,
        "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++   ",
        source.node.text,
        "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++   ",
    )

"""
Query
  ↓
Retriever
  ↓
Top 5 Nodes
  ↓
SimilarityPostprocessor
  ↓
Nodes >= 0.60
  ↓
Response Synthesizer
  ↓
LLM
"""