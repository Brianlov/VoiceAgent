"""
create_related_to.py
====================
Creates RELATED_TO relationships between Entity nodes that share a Context node.
Run this AFTER ingest_hotpotqa_graph.py.
"""

from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

load_dotenv()

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687"),
    auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "20010808"))
)
DB = os.getenv("NEO4J_DB", "hotpotqarag")

CYPHER = """
MATCH (a:Entity)-[:MENTIONED_IN]->(c:Context)<-[:MENTIONED_IN]-(b:Entity)
WHERE a <> b AND elementId(a) < elementId(b)
MERGE (a)-[:RELATED_TO]->(b)
RETURN count(*) AS created
"""

print("Creating RELATED_TO relationships (entities sharing a Context node)...")
with driver.session(database=DB) as session:
    result = session.run(CYPHER).single()
    print(f"  RELATED_TO relationships created: {result['created']}")

driver.close()
