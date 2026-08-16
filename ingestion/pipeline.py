# learn-llamaindex\ingestion\pipeline.py
import os

from llama_index.core import Settings
from llama_index.core.ingestion import (
    DocstoreStrategy,
    IngestionCache,
    IngestionPipeline,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.kvstore import SimpleKVStore
from llama_index.vector_stores.qdrant import QdrantVectorStore

PERSIST_DIR = "./pipeline_storage"

def create_ingestion_pipeline(
        vector_store: QdrantVectorStore,
) -> IngestionPipeline:
    """Creates a production-ready IngestionPipeline with persistent Caching & Deduplication."""

    # 1. Load or initialize persistent Docstore
    docstore_path = os.path.join(PERSIST_DIR, "docstore.json")
    if os.path.exists(docstore_path):
        docstore = SimpleDocumentStore.from_persist_dir(PERSIST_DIR)
    else:
        docstore = SimpleDocumentStore()

    # 2. Load or initialize persistent Cache
    cache_path = os.path.join(PERSIST_DIR, "cache.json")
    if os.path.exists(cache_path):
        cache = IngestionCache.from_persist_path(cache_path)
    else:
        cache = IngestionCache(cache=SimpleKVStore())

    return IngestionPipeline(
        transformations=[
            SentenceSplitter(
                chunk_size=256,
                chunk_overlap=20,
            ),
            Settings.embed_model,  # Generates vector embeddings for each node
        ],
        vector_store=vector_store,   # Automatically stores nodes + embeddings in Qdrant
        docstore=docstore,
        cache=cache,
        docstore_strategy=DocstoreStrategy.UPSERTS,
    )

def persist_pipeline_state(pipeline: IngestionPipeline) -> None:
    """Persists docstore and cache to disk so state survives application restarts."""
    os.makedirs(PERSIST_DIR, exist_ok=True)
    if pipeline.docstore:
        pipeline.docstore.persist(os.path.join(PERSIST_DIR, "docstore.json"))
    if pipeline.cache:
        pipeline.cache.persist(os.path.join(PERSIST_DIR, "cache.json"))