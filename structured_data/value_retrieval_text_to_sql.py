import os

from llama_index.core import Settings, SQLDatabase, VectorStoreIndex
from llama_index.core.indices.struct_store import SQLTableRetrieverQueryEngine
from llama_index.core.objects import (
    ObjectIndex,
    SQLTableNodeMapping,
    SQLTableSchema,
)
from llama_index.core.schema import TextNode
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from sqlalchemy import MetaData, create_engine, select, text

# 1. Models Setup
Settings.llm = Ollama(model="gemma4:26b", request_timeout=120.0)
Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text:latest",
    base_url="http://localhost:11434",
)

# 2. Initialize Database & Seed Categorical Values
db_path = "enterprise_advanced.db"
if os.path.exists(db_path):
    os.remove(db_path)

engine = create_engine(f"sqlite:///{db_path}")

with engine.begin() as conn:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY,
            customer_name TEXT,
            product TEXT,
            status TEXT,
            amount REAL,
            order_date TEXT
        )
    """))
    conn.execute(text("""
        INSERT INTO orders (customer_name, product, status, amount, order_date) VALUES
            ('Ali', 'Laptop', 'delivered', 1500, '2026-07-01'),
            ('Sara', 'Phone', 'processing', 900, '2026-07-03'),
            ('John', 'Monitor', 'delivered', 400, '2026-07-05'),
            ('Zainab', 'Laptop', 'shipped', 1700, '2026-07-08'),
            ('Bilal', 'Phone', 'cancelled', 900, '2026-07-10')
    """))

sql_database = SQLDatabase(engine)

# 3. Schema Retrieval Layer (ObjectIndex)
table_node_mapping = SQLTableNodeMapping(sql_database)
table_schema_objs = [
    SQLTableSchema(
        table_name="orders",
        context_str="Contains order transactions, product types, and order processing statuses."
    )
]
obj_index = ObjectIndex.from_objects(
    table_schema_objs,
    table_node_mapping,
    VectorStoreIndex,
)
table_retriever = obj_index.as_retriever(similarity_top_k=1)

# 4. Column Value Retrieval Layer
metadata = MetaData()
metadata.reflect(bind=engine)
orders_table = metadata.tables["orders"]

categorical_columns = ["status", "product"]
value_nodes = []

for col in categorical_columns:
    stmt = select(orders_table.c[col]).distinct()
    with engine.connect() as conn:
        distinct_vals = conn.execute(stmt).fetchall()
        
    for (val,) in distinct_vals:
        if val:
            # Store metadata linking value back to its table and column
            value_nodes.append(
                TextNode(
                    text=f"{col}: {val}",
                    metadata={"column_name": col, "value": str(val)}
                )
            )

value_index = VectorStoreIndex(value_nodes)
value_retriever = value_index.as_retriever(similarity_top_k=3)

# Map value retrievers by table name for SQLTableRetrieverQueryEngine
rows_retrievers = {
    "orders": value_retriever
}

# 5. Build Advanced SQL Engine
query_engine = SQLTableRetrieverQueryEngine(
    sql_database=sql_database,
    table_retriever=table_retriever,
    rows_retrievers=rows_retrievers,
    llm=Settings.llm,
)

# 6. Query Execution and Debug Inspection
if __name__ == "__main__":
    question = "How many orders have been successfully delivered?"

    print(f"\n==================== QUERY: '{question}' ====================")
    response = query_engine.query(question)

    print("\n[Synthesized Answer]:")
    print(response)

    print("\n[Generated SQL]:")
    print(response.metadata.get("sql_query"))

    print("\n[Raw Database Result]:")
    print(response.metadata.get("result"))