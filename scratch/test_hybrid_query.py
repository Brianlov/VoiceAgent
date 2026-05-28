import asyncio
from neo4j import AsyncGraphDatabase
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

load_dotenv()

async def test_query(query_text):
    uri = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    pwd = os.getenv("NEO4J_PASSWORD", "20010808")
    db = os.getenv("NEO4J_DB", "vectorchunkgraph")
    
    driver = AsyncGraphDatabase.driver(uri, auth=(user, pwd))
    model = SentenceTransformer('all-mpnet-base-v2')
    
    query_vector = model.encode(query_text).tolist()
    
    union_cypher = """
    // Part 1: Search Context vector index
    CALL db.index.vector.queryNodes('context_vector_index', 3, $query_vector) 
    YIELD node AS c, score AS context_score
    WITH c, context_score

    // Get entities mentioned in these contexts
    OPTIONAL MATCH (e_from_c:Entity)-[:MENTIONED_IN]->(c)
    WITH c, context_score, collect(DISTINCT e_from_c) AS entities_from_c

    // Get relationships for those entities
    UNWIND (case when size(entities_from_c) > 0 then entities_from_c else [null] end) AS e_c
    OPTIONAL MATCH (e_c)-[r_c]->(neighbor_c:Entity)
    WHERE type(r_c) <> "MENTIONED_IN" AND neighbor_c IN entities_from_c

    WITH c, context_score, 
         collect(DISTINCT coalesce(e_c.name, e_c.id) + " " + type(r_c) + " " + coalesce(neighbor_c.name, neighbor_c.id)) AS c_facts,
         collect(DISTINCT coalesce(e_c.name, e_c.id) + ": " + coalesce(e_c.description, "")) AS c_descs
         
    RETURN 
      c.text AS chunk_text,
      context_score AS final_score,
      c_facts AS all_facts,
      c_descs AS all_descriptions
    ORDER BY final_score DESC
    LIMIT 3

    UNION

    // Part 2: Search Entity vector index
    CALL db.index.vector.queryNodes('entity_vector_index', 3, $query_vector) 
    YIELD node AS e, score AS entity_score

    // Get relationships from these entities
    OPTIONAL MATCH (e)-[r]->(neighbor:Entity)
    WHERE type(r) <> "MENTIONED_IN"
    WITH e, entity_score, collect(DISTINCT coalesce(e.name, e.id) + " " + type(r) + " " + coalesce(neighbor.name, neighbor.id)) AS e_facts

    // Get contexts directly linked to these entities
    OPTIONAL MATCH (e)-[:MENTIONED_IN]->(c_direct:Context)
    WITH e, entity_score, e_facts, c_direct
    WHERE c_direct IS NOT NULL

    RETURN 
      c_direct.text AS chunk_text,
      entity_score AS final_score,
      e_facts AS all_facts,
      collect(DISTINCT coalesce(e.name, e.id) + ": " + coalesce(e.description, "")) AS all_descriptions
    ORDER BY final_score DESC
    LIMIT 3
    """
    
    print(f"\n==================================================")
    print(f"Testing Query: '{query_text}'")
    print(f"==================================================")
    
    async with driver.session(database=db) as session:
        result = await session.run(union_cypher, query_vector=query_vector)
        records = await result.data()
        
        print(f"Retrieved {len(records)} chunks:")
        for idx, r in enumerate(records):
            print(f"\n[{idx+1}] Score: {r['final_score']:.4f}")
            print(f"Text: {r['chunk_text'][:200]}...")
            if r['all_facts']:
                print(f"Facts: {r['all_facts']}")
            if r['all_descriptions']:
                # Filter out empty descriptions
                descs = [d for d in r['all_descriptions'] if d and not d.endswith(": ")]
                if descs:
                    print(f"Descriptions: {descs}")
                    
    await driver.close()

async def run_tests():
    queries = [
        "Where did Victoria spend the Christmas of 1900?",
        "In the 18th century there was one global trading hub for large diamonds, what was it?",
        "When did the Australian mandolin movement begin?",
        "What has excessive hunting contributed heavily to?"
    ]
    for q in queries:
        await test_query(q)

if __name__ == "__main__":
    asyncio.run(run_tests())
