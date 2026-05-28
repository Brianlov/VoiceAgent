import asyncio
from neo4j import AsyncGraphDatabase

async def test():
    driver = AsyncGraphDatabase.driver('bolt://127.0.0.1:7687', auth=('neo4j', '20010808'))
    async with driver.session(database='hotpotqarag') as session:
        res = await session.run("MATCH p=(e:Entity)-[r*1..2]-(n) WHERE e.name = 'Wild Wild West' AND ALL(rel IN r WHERE type(rel) <> 'MENTIONED_IN') RETURN p LIMIT 1")
        records = [record async for record in res]
        p = records[0]['p']
        print(type(p))
        print("Nodes:", len(p.nodes))
        for n in p.nodes:
            print("  Node:", dict(n))
        print("Relationships:", len(p.relationships))
        for r in p.relationships:
            print("  Rel type:", r.type)
    await driver.close()

asyncio.run(test())
