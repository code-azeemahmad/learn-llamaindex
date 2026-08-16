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

similarity_filter = SimilarityPostprocessor(
    similarity_cutoff=0.60,
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

retriever = index.as_retriever(
    similarity_top_k=5,
    filters=filters,
)

nodes = retriever.retrieve(
    "What is the deployment policy?"
)

for node in nodes:
    print(node.node.metadata)
    print(node.score)
    print(node.node.text)

query_engine = index.as_query_engine(
    similarity_top_k=20,
    filters=filters,
    node_postprocessors=[
        similarity_filter,
        reranker,
    ]
)

# Ask question
response = query_engine.query(
    "How does Retrieval Augmented Generation reduce hallucinations?"
)

print("\nAnswer:")
print(response)

# for source in response.source_nodes:
#     print( 
#         source.score,
#         source.node.text,
#     )
