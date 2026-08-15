import qdrant_client
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
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

# Create query engine
query_engine = index.as_query_engine(
    similarity_top_k=2
)

# Ask question
response = query_engine.query(
    "Why is chunking important in RAG?"
)

print("\nAnswer:")
print(response)

print("_" * 90)

collections = client.get_collections()
print(collections)

collection_info = client.get_collection(
    "llamaindex_rag"
)
print(collection_info)

print("_" * 90)

print(
    type(index.storage_context.vector_store)
)

print(
    index.storage_context.vector_store
)