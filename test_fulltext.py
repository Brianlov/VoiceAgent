import asyncio
from neo4j import AsyncGraphDatabase

async def test():
    driver = AsyncGraphDatabase.driver('bolt://127.0.0.1:7687', auth=('neo4j', '20010808'))
    async with driver.session(database='hotpotqarag') as session:
        print("Creating fulltext index...")
        await session.run('CREATE FULLTEXT INDEX entity_name_index IF NOT EXISTS FOR (n:Entity) ON EACH [n.name]')
        print("Index created. Querying 'Streptocitta piebalds'...")
        res = await session.run('CALL db.index.fulltext.queryNodes("entity_name_index", "Streptocitta piebalds") YIELD node AS e, score RETURN e.name, score LIMIT 5')
        data = [r.data() async for r in res]
        print("Results for Streptocitta:", data)
        
        print("Querying 'Victor Montana'...")
        res = await session.run('CALL db.index.fulltext.queryNodes("entity_name_index", "Victor Montana") YIELD node AS e, score RETURN e.name, score LIMIT 5')
        data = [r.data() async for r in res]
        print("Results for Victor Montana:", data)
        
    await driver.close()

asyncio.run(test())
