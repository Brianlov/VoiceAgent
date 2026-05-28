from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

load_dotenv()

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687"),
    auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "20010808"))
)
DB = os.getenv("NEO4J_DB", "hotpotqarag")

queries = {
    "Entity nodes":              "MATCH (e:Entity) RETURN count(e) AS n",
    "Context nodes":             "MATCH (c:Context) RETURN count(c) AS n",
    "Entities WITH embedding":   "MATCH (e:Entity) WHERE e.embedding IS NOT NULL RETURN count(e) AS n",
    "Contexts WITH embedding":   "MATCH (c:Context) WHERE c.embedding IS NOT NULL RETURN count(c) AS n",
    "MENTIONED_IN rels":         "MATCH ()-[:MENTIONED_IN]->() RETURN count(*) AS n",
    "RELATED_TO rels":           "MATCH ()-[:RELATED_TO]->() RETURN count(*) AS n",
    "Embedding dimension":       "MATCH (e:Entity) WHERE e.embedding IS NOT NULL RETURN size(e.embedding) AS n LIMIT 1",
}

with driver.session(database=DB) as session:
    for label, q in queries.items():
        result = session.run(q).single()
        val = result["n"] if result else "N/A"
        print(f"  {label:<30} {val}")

driver.close()
