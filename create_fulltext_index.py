from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

load_dotenv()

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687"),
    auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "20010808"))
)
DB = os.getenv("NEO4J_DB", "hotpotqarag")

with driver.session(database=DB) as session:
    session.run("""
        CREATE FULLTEXT INDEX entity_name_index IF NOT EXISTS
        FOR (n:Entity) ON EACH [n.name]
    """)
    print("Fulltext index entity_name_index created.")

driver.close()
