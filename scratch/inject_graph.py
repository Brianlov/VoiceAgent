import asyncio
from neo4j import AsyncGraphDatabase
from sentence_transformers import SentenceTransformer

async def inject_manual_knowledge():
    driver = AsyncGraphDatabase.driver('bolt://127.0.0.1:7687', auth=('neo4j', '20010808'))
    model = SentenceTransformer('all-mpnet-base-v2')
    
    # 1. Generate Embeddings for our new conceptual entities
    entities = {
        "Glucocorticoids": model.encode("Glucocorticoids").tolist(),
        "Hippocampus": model.encode("Hippocampus").tolist(),
        "Sleep": model.encode("Sleep").tolist()
    }
    
    async with driver.session(database='vectorchunkgraph') as session:
        # 2. Create Entities and set their embeddings
        for name, vector in entities.items():
            await session.run("""
                MERGE (e:Entity {id: $name})
                SET e.name = $name, e.embedding = $vector
            """, name=name, vector=vector)
            
        # 3. Create the logical Graph Relationships
        await session.run("""
            MATCH (gluco:Entity {id: 'Glucocorticoids'})
            MATCH (hippo:Entity {id: 'Hippocampus'})
            MATCH (sleep:Entity {id: 'Sleep'})
            MERGE (gluco)-[:DAMAGES]->(hippo)
            MERGE (sleep)-[:STABILIZES_MEMORIES_IN]->(hippo)
        """)
        
        # 4. Link the Entities to the actual Context chunks! 
        # (Using CONTAINS to find the exact chunks from your CSV)
        await session.run("""
            MATCH (c_stress:Context) WHERE c_stress.text CONTAINS 'Stressful life experiences may be a cause of memory loss'
            MATCH (c_sleep:Context) WHERE c_sleep.text CONTAINS 'Furthermore, some studies have shown that sleep deprivation'
            MATCH (gluco:Entity {id: 'Glucocorticoids'})
            MATCH (sleep:Entity {id: 'Sleep'})
            MATCH (hippo:Entity {id: 'Hippocampus'})
            
            MERGE (gluco)-[:MENTIONED_IN]->(c_stress)
            MERGE (hippo)-[:MENTIONED_IN]->(c_stress)
            
            MERGE (sleep)-[:MENTIONED_IN]->(c_sleep)
            MERGE (hippo)-[:MENTIONED_IN]->(c_sleep)
        """)
        
    print("✅ Successfully injected Sleep/Stress knowledge with embeddings!")
    await driver.close()

asyncio.run(inject_manual_knowledge())
