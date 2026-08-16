# learn-llamaindex\main.py
import qdrant_client
from bm25_index import BM25Index
from ingestion.pipeline import create_ingestion_pipeline, persist_pipeline_state
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
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
    model="llama3.1:8b",
    request_timeout=120.0,
)

Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text:latest",
    base_url="http://localhost:11434",
)

documents = SimpleDirectoryReader("data").load_data()
for doc in documents:
    # Ensuring stable document identity across runs
    if "file_name" in doc.metadata:
        doc.doc_id = doc.metadata["file_name"]
print(f"Loaded documents: {len(documents)}")

client = qdrant_client.QdrantClient(
    host="localhost",
    port=6333
)

vector_store = QdrantVectorStore(
    client=client,
    collection_name="llamaindex_rag",
)

index_store = SimpleIndexStore()


pipeline = create_ingestion_pipeline(
    vector_store=vector_store
)
nodes = pipeline.run(documents=documents)
print(f"Number of nodes: {len(nodes)}")

# Persist the cache and docstore to disk
persist_pipeline_state(pipeline)


docstore = SimpleDocumentStore()
docstore.add_documents(nodes)


storage_context = StorageContext.from_defaults(
    vector_store=vector_store,
    docstore=docstore,
    index_store=index_store,
)

print("\n========== STORAGE CONTEXT ==========")

print("Vector store:")
print(storage_context.vector_store)

print("\nDoc store:")
print(storage_context.docstore)

print("\nIndex store:")
print(storage_context.index_store)


print("\n========== VECTOR STORE ==========")

print("Type:")
print(type(vector_store))

print("\nCollection:")
print(vector_store.collection_name)

print("\nClient:")
print(type(vector_store.client))


print("\n========== DOCSTORE ==========")

print(type(storage_context.docstore))

print("Number of stored nodes:")
print(len(storage_context.docstore.docs))

node_id = nodes[0].node_id
print("Node ID:", node_id)

stored_node = storage_context.docstore.get_document(
    node_id
)
print("\nStored node:")
print(stored_node.text)


print("\n========== INDEX STORE ==========")

print(
    "Type:",
    type(storage_context.index_store)
)

print(
    "Index structures:",
    storage_context.index_store.index_structs()
)

index_structs = storage_context.index_store.index_structs()

# print("\nIndex ID:")
# print(index.index_id)

for struct in index_structs:
    print("\n========== INDEX STRUCT ==========")
    print(type(struct))
    print(struct)

# print("\n========== Persistence ==========")
# index.storage_context.persist(
#     persist_dir="./storage"
# )

# bm25_index = BM25Index()
# bm25_index.add_documents(nodes)
# bm25_retriever = BM25Retriever(
#     bm25_index=bm25_index,
#     similarity_top_k=5,
# )

# similarity_filter = SimilarityPostprocessor(
#     similarity_cutoff=0.30,
# )

# # Reranker after retrieval
# reranker = SentenceTransformerRerank(
#     model="BAAI/bge-reranker-base",
#     top_n=3,
# )

# # Metadata Filtering
# filters = MetadataFilters(
#     filters=[
#         MetadataFilter(
#             key="category",
#             value="rag",
#         ),
#         MetadataFilter(
#             key="document_type",
#             value="policy",
#         )
#     ]
# )

# Dense Vector Index (Load directly from vector store to avoid re-insertion)
# index = VectorStoreIndex.from_vector_store(vector_store)

# base_retriever = index.as_retriever(
#     similarity_top_k=5,
# )


# # QueryFusionRetriever Abstraction
# fusion_retriever = QueryFusionRetriever(
#     retrievers=[
#         base_retriever,
#         bm25_retriever,
#     ],
#     llm=Settings.llm,
#     similarity_top_k=5,
#     num_queries=4,
#     mode=FUSION_MODES.RECIPROCAL_RANK,
#     use_async=False,
#     verbose=True,
# )

query = "How can RAG improves Enterprise AI applications?"


# query_engine = RetrieverQueryEngine.from_args(
#     retriever=fusion_retriever,
#     node_postprocessors=[
#         # similarity_filter,
#         reranker,
#     ],
# )


# response = query_engine.query(
#     query
# )

# print("\nAnswer:")
# print(response)

# for source in response.source_nodes:
#     print( 
#         source.score,
#         source.node.text,
#     )
