# learn-llamaindex\main.py
import qdrant_client
from llama_index.core import (
    # QueryBundle,  
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
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

# from query_rewriter import QueryRewriter

llm = Settings.llm = Ollama(
    model="gemma4:26b",
    request_timeout=120.0,
)

Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text:latest",
    base_url="http://localhost:11434",
)

documents = SimpleDirectoryReader("data").load_data()
print(f"Loaded documents: {len(documents)}")

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
    similarity_cutoff=0.30,
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



'''
original_query="How can RAG improve enterprise AI applications?"
rewriter = QueryRewriter(llm)

queries = rewriter.rewrite(
    original_query,
)

print("Generated queries:")
for q in queries:
    print(" -", q)
'''

# Build index
index = VectorStoreIndex(
    nodes, 
    storage_context=storage_context
)

base_retriever = index.as_retriever(
    similarity_top_k=5,
)


'''Manual Version
# Step 1: retrieve for each query separately
nodes_lists = []
for q in queries:
    results = retriever.retrieve(q)
    print(f"\nQuery: {q}")
    for n in results:
        print(round(n.score, 4), n.node.text[:100])
    nodes_lists.append(results)


# Step 2: Reciprocal Rank Fusion
def reciprocal_rank_fusion(nodes_lists, k: int = 60):
    scores = {}
    node_lookup = {}

    for results in nodes_lists:
        for rank, node in enumerate(results):
            node_id = node.node.node_id
            node_lookup[node_id] = node
            scores[node_id] = scores.get(node_id, 0) + 1 / (k + rank + 1)

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(node_lookup[nid], score) for nid, score in fused]


fused_nodes = reciprocal_rank_fusion(nodes_lists)

print("\n--- Fused results (RRF) ---")
for node, score in fused_nodes:
    print(round(score, 4), node.node.text[:100])

fused_nodes = [n for n, _ in fused_nodes]

reranked_nodes = reranker.postprocess_nodes(
    fused_nodes,
    query_bundle=QueryBundle(query_str=original_query),
)

for n in reranked_nodes:
    print(n.score, n.node.text[:100])
'''


# QueryFusionRetriever Abstraction
fusion_retriever = QueryFusionRetriever(
    retrievers=[base_retriever],
    llm=Settings.llm,
    similarity_top_k=5,
    num_queries=3,
    mode=FUSION_MODES.RECIPROCAL_RANK,
    use_async=False,
    verbose=True,
)

query = "What is Retrieval-Augmented Generation?"

# nodes = fusion_retriever.retrieve(query)    

# print("\n========== FUSED RESULTS ==========")

# for i, node in enumerate(nodes):
#     print(f"\n--- Node {i} ---")
#     print("Score:", node.score)
#     print("Text:")
#     print(node.node.text)

query_engine = RetrieverQueryEngine.from_args(
    retriever=fusion_retriever,
    node_postprocessors=[
        reranker,
        similarity_filter,
    ],
)


response = query_engine.query(
    query
)

print("\nAnswer:")
print(response)

# for source in response.source_nodes:
#     print( 
#         source.score,
#         source.node.text,
#     )


"""
node_postprocessors execute sequentially in a pipeline from left to right (index 0 to index N). The output of postprocessor #1 becomes the input for postprocessor #2.
"""
"""node_postprocessors = [
    reranker,          # 1. Update scores using cross-encoder
    similarity_filter, # 2. Filter out weak scores
]"""
# Scenario A: [similarity_filter, reranker] --> Empty Response
# Scenario B: [reranker, similarity_filter] --> Gives Answer

