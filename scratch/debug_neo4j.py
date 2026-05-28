import asyncio
from neo4j import AsyncGraphDatabase

async def run():
    driver = AsyncGraphDatabase.driver('bolt://127.0.0.1:7687', auth=('neo4j', '20010808'))
    async with driver.session(database='vectorchunkgraph') as session:
        result = await session.run('MATCH (n:Entity) RETURN n LIMIT 1')
        records = await result.data()
        print('Entity nodes:', records)
    await driver.close()

if __name__ == '__main__':
    asyncio.run(run())
