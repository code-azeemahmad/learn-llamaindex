# learn-llamaindex\ingestion\pipeline.py
from llama_index.core import Settings
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.qdrant import QdrantVectorStore


def create_ingestion_pipeline(
        vector_store: QdrantVectorStore,
) -> IngestionPipeline:

    return IngestionPipeline(
        transformations=[
            SentenceSplitter(
                chunk_size=256,
                chunk_overlap=20,
            ),
            Settings.embed_model,  # Generates vector embeddings for each node
        ],
        vector_store=vector_store   # Automatically stores nodes + embeddings in Qdrant
    )