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
        # Check total Context nodes
        res1 = await session.run("MATCH (n:Context) RETURN count(n) AS c")
        c_count = (await res1.single())["c"]
        
        # Check total Entity nodes
        res2 = await session.run("MATCH (n:Entity) RETURN count(n) AS c")
        e_count = (await res2.single())["c"]
        
        # Check total relationships
        res3 = await session.run("MATCH ()-[r]->() RETURN count(r) AS c")
        r_count = (await res3.single())["c"]
        
        print(f"Total Context nodes: {c_count}")
        print(f"Total Entity nodes: {e_count}")
        print(f"Total Relationships: {r_count}")
        
    await driver.close()

if __name__ == "__main__":
    asyncio.run(main())
