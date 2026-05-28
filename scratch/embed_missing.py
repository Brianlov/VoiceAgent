import os
import asyncio
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687"),
    auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "20010808"))
)
DB = os.getenv("NEO4J_DB", "hotpotqarag")

print("Loading embedding model...")
model = SentenceTransformer("all-mpnet-base-v2")

def embed_missing():
    with driver.session(database=DB) as session:
        # Context nodes missing embeddings
        result = session.run("MATCH (c:Context) WHERE c.embedding IS NULL RETURN elementId(c) as id, c.text as text")
        records = list(result)
        if records:
            print(f"Embedding {len(records)} missing Context nodes...")
            texts = [r["text"] for r in records]
            embs = model.encode(texts).tolist()
            for r, emb in zip(records, embs):
                session.run("MATCH (c:Context) WHERE elementId(c) = $id SET c.embedding = $emb", id=r["id"], emb=emb)
        else:
            print("No Context nodes missing embeddings.")

        # Entity nodes missing embeddings
        result = session.run("MATCH (e:Entity) WHERE e.embedding IS NULL RETURN elementId(e) as id, coalesce(e.description, coalesce(e.name, '')) as text")
        records = list(result)
        if records:
            print(f"Embedding {len(records)} missing Entity nodes...")
            texts = [r["text"] for r in records]
            embs = model.encode(texts).tolist()
            for r, emb in zip(records, embs):
                session.run("MATCH (e:Entity) WHERE elementId(e) = $id SET e.embedding = $emb", id=r["id"], emb=emb)
        else:
            print("No Entity nodes missing embeddings.")

embed_missing()
driver.close()
print("Done embedding missing nodes.")
