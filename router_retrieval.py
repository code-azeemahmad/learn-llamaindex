# learn-llamaindex\router_retrieval.py
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import RouterRetriever
from llama_index.core.tools import RetrieverTool
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.retrievers.bm25 import BM25Retriever

# 1. Configure Global Models
Settings.llm = Ollama(
    model="gemma4:26b",
    request_timeout=120.0,
)

Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text:latest",
    base_url="http://localhost:11434",
)

# 2. Load Documents Directly
documents = SimpleDirectoryReader("data").load_data()

# 3. Construct In-Memory Indexes & Retrievers
vector_index = VectorStoreIndex.from_documents(documents)
dense_retriever = vector_index.as_retriever(similarity_top_k=5)

# Pass nodes directly to BM25 to avoid cache desynchronization
bm25_retriever = BM25Retriever.from_defaults(
    nodes=vector_index.docstore.docs.values(),
    similarity_top_k=5,
)

# 4. Wrap Retrievers in RetrieverTools
dense_tool = RetrieverTool.from_defaults(
    retriever=dense_retriever,
    description=(
        "Use this retriever for semantic, conceptual, and broad questions "
        "where understanding meaning and context is essential."
    ),
)

bm25_tool = RetrieverTool.from_defaults(
    retriever=bm25_retriever,
    description=(
        "Use this retriever for exact keywords, error codes, technical terms, "
        "identifiers, exact names, and phrase matching."
    ),
)

# 5. Build Standalone RouterRetriever
router = RouterRetriever.from_defaults(
    retriever_tools=[dense_tool, bm25_tool],
    llm=Settings.llm,
    select_multi=False,
)

# 6. Execute Retrieval & Synthesize Response
query = "What is ERR-401?"

print(f"\n==================== ROUTING QUERY: '{query}' ====================")
nodes = router.retrieve(query)

for i, result in enumerate(nodes):
    print(f"\n--- Retrieved Node {i+1} ---")
    print("Score:", result.score)
    print("Text:", result.node.text[:250] + "...")

# 7. Query Engine Synthesis
query_engine = RetrieverQueryEngine.from_args(
    retriever=router,
    llm=Settings.llm,
)

response = query_engine.query(query)

print("\n==================== SYNTHESIZED ANSWER ====================")
print(response)


# When Python executes an import statement, it runs the imported file from top to bottomz       
"""
BM25 scores in LlamaIndex are raw sparse relevance scores (often numbers greater than 1.0, such as 2.51, 2.29).
Dense similarity scores (like Cosine) range strictly between -1.0 and 1.0 (typically 0.1 to 0.8).
"""