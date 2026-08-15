from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    VectorStoreIndex,
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
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
documents = SimpleDirectoryReader("data").load_data()

print(f"Loaded documents: {len(documents)}")

print("First document metadata:")
print(documents[0].metadata)

pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(
            chunk_size=256,
            chunk_overlap=20
        ),
        # Additional transformations
    ]
)

nodes = pipeline.run(documents=documents)

print(f"\nNumber of nodes: {len(nodes)}")

for i, node in enumerate(nodes):
    print(f"\n--- Node {i} ---")
    print(f"Node ID: {node.node_id}")
    print(f"Text: {node.text}")
    print(f"Metadata: {node.metadata}")

# Build index
index = VectorStoreIndex.from_documents(nodes)

# Inspect the nodes
nodes = list(index.docstore.docs.values())  # inspection/debugging technique, not used in production

# Create query engine
query_engine = index.as_query_engine()

# Ask question
response = query_engine.query(
    "Why is chunking important in RAG?"
)

print("\nAnswer:")
print(response)