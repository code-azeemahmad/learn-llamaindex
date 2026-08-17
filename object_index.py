from llama_index.core import (
    Settings,
    VectorStoreIndex,
)
from llama_index.core.objects import ObjectIndex
from llama_index.core.tools import FunctionTool
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

# 1. SETUP GLOBAL MODELS
Settings.llm = Ollama(
    model="gemma4:26b",
    request_timeout=120.0,
)

Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text:latest",
    base_url="http://localhost:11434",
)

def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


add_tool = FunctionTool.from_defaults(
    fn=add,
    name="add",
)

multiply_tool = FunctionTool.from_defaults(
    fn=multiply,
    name="multiply",
)


tools = [
    add_tool,
    multiply_tool,
]


obj_index = ObjectIndex.from_objects(
    tools,
    index_cls=VectorStoreIndex,
)


obj_retriever = obj_index.as_retriever(
    similarity_top_k=1,
)


query = "Which tool can multiply two numbers?"

objects = obj_retriever.retrieve(query)


print("\n========== RETRIEVED OBJECTS ==========")

for obj in objects:
    print("Type:", type(obj))
    print("Name:", obj.metadata.name)
    print("Description:", obj.metadata.description)