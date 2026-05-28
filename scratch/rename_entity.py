from neo4j import GraphDatabase
import os

driver = GraphDatabase.driver(
    os.getenv('NEO4J_URI', 'bolt://127.0.0.1:7687'), 
    auth=(os.getenv('NEO4J_USER', 'neo4j'), os.getenv('NEO4J_PASSWORD', '20010808'))
)
with driver.session(database=os.getenv('NEO4J_DB', 'hotpotqarag')) as session:
    session.run('MATCH (e:Entity {name: "Acolytes Protection Agency"}) SET e.name = "Acolytes Protection Agency (APA)"')
    # Update the full text index
    session.run('DROP INDEX entity_name_index IF EXISTS')
    session.run('CREATE FULLTEXT INDEX entity_name_index FOR (n:Entity) ON EACH [n.name]')

driver.close()
