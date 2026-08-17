from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    VectorStoreIndex,
)
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.schema import IndexNode
from llama_index.embeddings.ollama import OllamaEmbedding

Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text:latest",
    base_url="http://localhost:11434",
)

# 1. Load sub-domain documents
rag_documents = SimpleDirectoryReader(input_files=["data/rag.txt"]).load_data()
python_documents = SimpleDirectoryReader(input_files=["data/python.txt"]).load_data()

# 2. Build sub-indexes and sub-retrievers
rag_index = VectorStoreIndex.from_documents(rag_documents)
python_index = VectorStoreIndex.from_documents(python_documents)

rag_retriever = rag_index.as_retriever(similarity_top_k=2)
python_retriever = python_index.as_retriever(similarity_top_k=2)

# 3. Create pointer nodes (IndexNodes) that reference sub-retrievers
nodes = [
    IndexNode(
        text="Comprehensive information about Retrieval-Augmented Generation (RAG), vector stores, and embeddings.",
        index_id="rag_retriever",
    ),
    IndexNode(
        text="Concepts, syntax, data structures, and programming patterns in Python.",
        index_id="python_retriever",
    ),
]

# 4. Build root index over the IndexNodes
root_index = VectorStoreIndex(nodes)
root_retriever = root_index.as_retriever(similarity_top_k=1)

# 5. Map index_ids to their target objects
retriever_dict = {
    "rag_retriever": rag_retriever,
    "python_retriever": python_retriever,
}

# 6. Instantiate RecursiveRetriever
recursive_retriever = RecursiveRetriever(
    root_id="root",
    retriever_dict={"root": root_retriever, **retriever_dict},
    verbose=True,
)

# 7. Execute test query
query = "How does chunking affect context precision in RAG?"
print(f"\n==================== QUERY: '{query}' ====================")

results = recursive_retriever.retrieve(query)

print(f"\nRetrieved {len(results)} target nodes:")
for i, res in enumerate(results):
    print(f"\n--- Result #{i+1} ---")
    print(f"Node ID:      {res.node.node_id}")
    print(f"Text Snippet: {res.node.text[:200]}...")