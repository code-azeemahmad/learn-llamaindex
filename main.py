from llama_index.core import (  # two core abstractions
    Settings,
    SimpleDirectoryReader,
    VectorStoreIndex,
)
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

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
documents = SimpleDirectoryReader("data").load_data()   # Load documents from this directory.

print(f"Loaded documents: {len(documents)}")

print("First document metadata:")
print(documents[0].metadata)


# Build index
index = VectorStoreIndex.from_documents(documents)  # Documents are transformed into nodes

# Create query engine
query_engine = index.as_query_engine()  # Give a query interface over this index

# Ask question
response = query_engine.query(  # LlamaIndex orchestrates the retrieval/query process
    "What is RAG?"
)

print("\nAnswer:")
print(response)