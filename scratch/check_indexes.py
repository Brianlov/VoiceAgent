import asyncio
from neo4j import AsyncGraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

async def main():
    uri = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    pwd = os.getenv("NEO4J_PASSWORD", "20010808")
    db = os.getenv("NEO4J_DB", "vectorchunkgraph")
    
    print(f"Connecting to database '{db}'...")
    driver = AsyncGraphDatabase.driver(uri, auth=(user, pwd))
    
    async with driver.session(database=db) as session:
        # Show all indexes
        res = await session.run("SHOW INDEXES")
        records = await res.data()
        
        print("\nExisting Indexes in Neo4j:")
        for r in records:
            print(f"- Name: {r.get('name')}, Type: {r.get('type')}, State: {r.get('state')}, LabelsOrTypes: {r.get('labelsOrTypes')}, Properties: {r.get('properties')}")
            
    await driver.close()

if __name__ == "__main__":
    asyncio.run(main())
