import asyncio
from neo4j import AsyncGraphDatabase

async def find_entities():
    driver = AsyncGraphDatabase.driver('bolt://127.0.0.1:7687', auth=('neo4j', '20010808'))
    terms = ['Scholastic', 'Notre Dame', 'Virgin Mary', 'Glucocorticoid', 'Hippocamp', 'Sleep', 'Doctorate', 'Hunting', 'Extinction', 'Henry Art', 'Seattle', 'Mass transit', 'Regal', 'British India']
    
    async with driver.session(database='vectorchunkgraph') as session:
        for term in terms:
            query = """
            MATCH (e:Entity)
            WHERE toLower(e.id) CONTAINS toLower($term) OR toLower(e.name) CONTAINS toLower($term)
            RETURN e.id AS id LIMIT 3
            """
            result = await session.run(query, term=term)
            records = await result.data()
            print(f"Term '{term}': {[r['id'] for r in records]}")
            
    await driver.close()

asyncio.run(find_entities())
