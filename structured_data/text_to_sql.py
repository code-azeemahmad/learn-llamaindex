import os

from llama_index.core import Settings, SQLDatabase, VectorStoreIndex
from llama_index.core.indices.struct_store import SQLTableRetrieverQueryEngine
from llama_index.core.objects import (
    ObjectIndex,
    SQLTableNodeMapping,
    SQLTableSchema,
)
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from sqlalchemy import create_engine, text

# 1. Setup Models
Settings.llm = Ollama(
    model="gemma4:26b",
    request_timeout=120.0,
)
Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text:latest",
    base_url="http://localhost:11434",
)

# 2. Initialize Database & Seed Schema
db_path = "enterprise.db"
if os.path.exists(db_path):
    os.remove(db_path)

engine = create_engine(f"sqlite:///{db_path}")

with engine.begin() as connection:
    connection.execute(text("""
        CREATE TABLE IF NOT EXISTS customers (
            id INTEGER PRIMARY KEY,
            name TEXT,
            country TEXT
        )
    """))
    connection.execute(text("""
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER,
            product TEXT,
            amount REAL,
            order_date TEXT,
            FOREIGN KEY(customer_id) REFERENCES customers(id)
        )
    """))
    connection.execute(text("""
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY,
            product TEXT,
            rating INTEGER,
            review_text TEXT
        )
    """))

    # Seed Sample Data
    connection.execute(text("""
        INSERT INTO customers (id, name, country) VALUES
            (1, 'Ali', 'Pakistan'),
            (2, 'Sara', 'Pakistan'),
            (3, 'John', 'USA')
    """))
    connection.execute(text("""
        INSERT INTO orders (customer_id, product, amount, order_date) VALUES
            (1, 'Laptop', 1500, '2026-07-01'),
            (2, 'Phone', 900, '2026-07-03'),
            (1, 'Monitor', 400, '2026-07-05'),
            (3, 'Laptop', 1700, '2026-07-08')
    """))
    connection.execute(text("""
        INSERT INTO reviews (product, rating, review_text) VALUES
            ('Laptop', 5, 'Excellent build quality.'),
            ('Phone', 4, 'Good battery life.')
    """))

# 3. Create LlamaIndex SQLDatabase Abstraction
sql_database = SQLDatabase(engine)

# 4. Map Tables to SQLTableSchema Objects
table_node_mapping = SQLTableNodeMapping(sql_database)
table_names = sql_database.get_usable_table_names()

table_schema_objs = [
    SQLTableSchema(
        table_name=name,
        context_str=(
            "Contains customer demographic data." if name == "customers"
            else "Contains transaction orders with customer_id and amounts in USD." if name == "orders"
            else "Contains product reviews and star ratings."
        )
    )
    for name in table_names
]

# 5. Build ObjectIndex for Dynamic Table Schema Retrieval
obj_index = ObjectIndex.from_objects(
    table_schema_objs,
    table_node_mapping,
    VectorStoreIndex,
)

table_retriever = obj_index.as_retriever(similarity_top_k=2)

# 6. Build SQLTableRetrieverQueryEngine
query_engine = SQLTableRetrieverQueryEngine(
    sql_database=sql_database,
    table_retriever=table_retriever,
    llm=Settings.llm,
)

# 7. Execute Test Queries & Inspect Results
if __name__ == "__main__":
    question = "Which customer spent the most total money?"

    print(f"\n==================== QUERY: '{question}' ====================")
    response = query_engine.query(question)

    print("\n[Synthesized Answer]:")
    print(response)

    print("\n[Generated SQL]:")
    print(response.metadata.get("sql_query"))

    print("\n[Raw Database Output]:")
    print(response.metadata.get("result"))