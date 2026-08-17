from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import (
    HierarchicalNodeParser,
    get_leaf_nodes,
    get_root_nodes,
)
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

llm = Settings.llm = Ollama(
    model="llama3.1:8b",
    request_timeout=120.0,
)

Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text:latest",
    base_url="http://localhost:11434",
)

documents = SimpleDirectoryReader(
    "data"
).load_data()


node_parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[512, 256, 128],
)


nodes = node_parser.get_nodes_from_documents(
    documents
)


leaf_nodes = get_leaf_nodes(nodes)
root_nodes = get_root_nodes(nodes)


print("Total nodes:", len(nodes))
print("Root nodes:", len(root_nodes))
print("Leaf nodes:", len(leaf_nodes))


docstore = SimpleDocumentStore()
docstore.add_documents(nodes)

storage_context = StorageContext.from_defaults(
    docstore=docstore,
)

index = VectorStoreIndex(
    leaf_nodes,
    storage_context=storage_context,
)

base_retriever = index.as_retriever(
    similarity_top_k=6,
)

retriever = AutoMergingRetriever(
    base_retriever,
    storage_context,
    verbose=True,
)

query = (
    "What are the potential effects of increasing "
    "the amount of safety data used during RLHF?"
)

base_nodes = base_retriever.retrieve(query)

merged_nodes = retriever.retrieve(query)

print("\n========== BASE RETRIEVER ==========")

for node in base_nodes:
    print("\nScore:", node.score)
    print("ID:", node.node.node_id)
    print("Text:", node.node.text[:300])

print("\n========== AUTO MERGING ==========")

for node in merged_nodes:
    print("\nScore:", node.score)
    print("ID:", node.node.node_id)
    print("Text:", node.node.text[:500])