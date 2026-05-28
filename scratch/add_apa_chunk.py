from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

load_dotenv()

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687"),
    auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "20010808"))
)
DB = os.getenv("NEO4J_DB", "hotpotqarag")

cypher = """
MERGE (c:Context {chunk_id: "manual__APA_wrestling"})
SET c.text = "The Acolytes Protection Agency (APA) was a professional wrestling tag team that consisted of Bradshaw (John Layfield) and Faarooq (Ron Simmons). They wrestled for WWF/WWE between October 1998 and March 2004."
MERGE (e:Entity {name: "Acolytes Protection Agency"})
SET e.description = "WWE professional wrestling tag team, members: Bradshaw and Faarooq (Ron Simmons)"
MERGE (e)-[:MENTIONED_IN]->(c)
WITH c
MATCH (rs:Entity {name: "Ron Simmons"})
MERGE (rs)-[:MENTIONED_IN]->(c)
"""

with driver.session(database=DB) as session:
    session.run(cypher)
    print("Added APA wrestling manual chunk.")

driver.close()
