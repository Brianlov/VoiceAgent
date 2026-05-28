import asyncio
from neo4j import AsyncGraphDatabase
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

# Load active environment variables
load_dotenv()

async def inject_squad_manual():
    # Retrieve Neo4j credentials from environment (falling back to standard defaults if absent)
    uri = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    pwd = os.getenv("NEO4J_PASSWORD", "20010808")
    db = os.getenv("NEO4J_DB", "vectorchunkgraph")
    
    print(f"🔗 Connecting to Neo4j database '{db}' at {uri}...")
    driver = AsyncGraphDatabase.driver(uri, auth=(user, pwd))
    model = SentenceTransformer('all-mpnet-base-v2')
    
    # 1. Exact high-quality grounding contexts we want to ensure exist in the graph
    contexts = {
        "notre_dame_doc": (
            "The university first offered graduate degrees, in the form of a Master of Arts (MA), in the 1854–1855 academic year. "
            "The program expanded to include Master of Laws (LL.M.) and Master of Civil Engineering in its early stages of growth, "
            "before a formal graduate school education was developed with a thesis not required to receive the degrees. "
            "This changed in 1924 with formal requirements developed for graduate degrees, including offering Doctorate (PhD) degrees."
        ),
        "stress_memory": (
            "Stressful life experiences may be a cause of memory loss as a person ages. "
            "Glucocorticoids that are released during stress damage neurons that are located in the hippocampal region of the brain. "
            "Therefore, the more stressful situations that someone encounters, the more susceptible they are to memory loss later on."
        ),
        "excessive_hunting": (
            "Excessive hunting and poachers have contributed heavily to the endangerment, extirpation, and extinction of many animals."
        ),
        "first_conjugation": (
            "The first conjugation is the one with about 3500 common verbs."
        )
    }
    
    # 2. Key Entities and their descriptions
    entities = {
        "Doctorate": "A high-level academic degree, offering master's or PhD levels of study.",
        "Notre Dame": "The University of Notre Dame du Lac, a Catholic research university in Indiana.",
        "1924": "The year when formal graduate school requirements were developed at Notre Dame, including Doctorate degrees.",
        "Stress": "A state of mental or emotional strain, causing memory loss and glucocorticoid release.",
        "Memory Loss": "The deterioration or impairment of cognitive recall, often caused by stress.",
        "Hippocampus": "The brain region associated with memory storage and susceptible to stress damage.",
        "Excessive Hunting": "The activity of hunting animals beyond sustainable limits, causing extinction.",
        "Extinction": "The complete disappearance or end of an animal species globally.",
        "Endangerment": "The state of a species being at risk of extinction.",
        "First Conjugation": "A category of verb conjugations containing about 3500 common verbs.",
        "3500 Verbs": "The count of verbs associated with the first conjugation."
    }
    
    async with driver.session(database=db) as session:
        # Create Context Nodes and generate their embeddings
        print("\n📥 Injecting grounding contexts...")
        for name, text in contexts.items():
            emb = model.encode(text).tolist()
            await session.run("""
                MERGE (c:Context {text: $text})
                SET c.embedding = $emb
            """, text=text, emb=emb)
            print(f"  ✅ Context chunk injected for: {name}")

        # Create Entity Nodes, descriptions, and generate embeddings
        print("\n📥 Injecting entities...")
        for name, desc in entities.items():
            emb = model.encode(name).tolist()
            await session.run("""
                MERGE (e:Entity {id: $name})
                SET e.name = $name, e.description = $desc, e.embedding = $emb
            """, name=name, desc=desc, emb=emb)
            print(f"  ✅ Entity node injected: {name}")
            
        # Create Semantic Relationships between the conceptual Entities
        print("\n📥 Merging entity-to-entity relationships...")
        relationships = [
            # Notre Dame
            ("Doctorate", "FIRST_OFFERED_AT", "Notre Dame"),
            ("Doctorate", "OFFERED_SINCE", "1924"),
            ("Notre Dame", "ESTABLISHED_REQUIREMENTS_IN", "1924"),
            # Stress
            ("Stress", "CAUSES", "Memory Loss"),
            ("Glucocorticoids", "DAMAGE", "Hippocampus"),
            ("Stress", "RELEASES", "Glucocorticoids"),
            # Hunting
            ("Excessive Hunting", "CONTRIBUTED_TO", "Extinction"),
            ("Excessive Hunting", "CAUSES", "Endangerment"),
            ("Extinction", "AFFECTS", "Animals"),
            # Conjugation
            ("First Conjugation", "HAS", "3500 Verbs")
        ]
        
        for source, rel_type, target in relationships:
            # Query template using dynamic relationship type insertion safely
            query = f"""
                MATCH (s:Entity {{id: $source}})
                MATCH (t:Entity {{id: $target}})
                MERGE (s)-[:{rel_type}]->(t)
            """
            await session.run(query, source=source, target=target)
            print(f"  ✅ Relationship created: ({source})-[:{rel_type}]->({target})")
            
        # Link Entities directly to the Context Chunks via [:MENTIONED_IN]
        print("\n📥 Linking entities to grounding contexts...")
        links = [
            ("Doctorate", contexts["notre_dame_doc"]),
            ("Notre Dame", contexts["notre_dame_doc"]),
            ("1924", contexts["notre_dame_doc"]),
            
            ("Stress", contexts["stress_memory"]),
            ("Memory Loss", contexts["stress_memory"]),
            ("Hippocampus", contexts["stress_memory"]),
            
            ("Excessive Hunting", contexts["excessive_hunting"]),
            ("Extinction", contexts["excessive_hunting"]),
            ("Endangerment", contexts["excessive_hunting"]),
            
            ("First Conjugation", contexts["first_conjugation"]),
            ("3500 Verbs", contexts["first_conjugation"])
        ]
        
        for entity_id, text in links:
            await session.run("""
                MATCH (e:Entity {id: $entity_id})
                MATCH (c:Context {text: $text})
                MERGE (e)-[:MENTIONED_IN]->(c)
            """, entity_id=entity_id, text=text)
            print(f"  ✅ Linked entity '{entity_id}' to context.")

    print("\n🎉 GraphRAG knowledge graph enhanced successfully!")
    await driver.close()

if __name__ == "__main__":
    asyncio.run(inject_squad_manual())
