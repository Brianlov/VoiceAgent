"""
ingest_hotpotqa_graph.py
========================
Reads hotpotqa_sampled_100.csv and indexes into Neo4j (hotpotqarag) with:

  - Context nodes  : one per Wikipedia paragraph (text + embedding)
  - Entity nodes   : one per article title (name + embedding of the paragraph)
  - MENTIONED_IN   : (Entity)-[:MENTIONED_IN]->(Context)
  - RELATED_TO     : (Entity)-[:RELATED_TO]->(Entity) for entities that share
                     the same HotpotQA question (co-mention = multi-hop bridge)

You can later add custom relationships manually via Neo4j Browser.

Usage:
  .venv\\Scripts\\python.exe ingest_hotpotqa_graph.py
  .venv\\Scripts\\python.exe ingest_hotpotqa_graph.py --csv hotpotqa_sampled_100.csv --db hotpotqarag --batch 32
"""

import argparse
import ast
import os
import re
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

load_dotenv()

# ── defaults ──────────────────────────────────────────────────────────────────
NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://127.0.0.1:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "20010808")
NEO4J_DB       = os.getenv("NEO4J_DB",       "hotpotqarag")
MODEL_NAME     = "all-mpnet-base-v2"


# ── helpers ───────────────────────────────────────────────────────────────────

def safe_parse(value):
    """Parse the ugly numpy-repr strings that pandas stores for the context column."""
    if not isinstance(value, str):
        return None
    try:
        # Replace numpy-specific tokens so ast.literal_eval can handle it
        cleaned = re.sub(r"array\(", "[", value)
        cleaned = re.sub(r",\s*dtype=\w+\)", "]", cleaned)
        return ast.literal_eval(cleaned)
    except Exception:
        return None


def extract_paragraphs(context_value):
    """
    Returns list of (title: str, paragraph_text: str).
    Each article contributes ONE paragraph = all its sentences joined.
    """
    parsed = safe_parse(context_value)
    if parsed is None or not isinstance(parsed, dict):
        return []

    titles     = list(parsed.get("title", []))
    sentences  = list(parsed.get("sentences", []))

    paragraphs = []
    for i, title in enumerate(titles):
        if i >= len(sentences):
            break
        sents = list(sentences[i]) if hasattr(sentences[i], "__iter__") and not isinstance(sentences[i], str) else [sentences[i]]
        text  = " ".join(str(s).strip() for s in sents if str(s).strip())
        if text:
            paragraphs.append((str(title).strip(), text))
    return paragraphs


def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


# ── Neo4j setup ───────────────────────────────────────────────────────────────

SETUP_CYPHER = [
    # Uniqueness constraints
    "CREATE CONSTRAINT entity_name_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
    "CREATE CONSTRAINT context_id_unique IF NOT EXISTS FOR (c:Context) REQUIRE c.chunk_id IS UNIQUE",

    # Vector index on Entity embeddings (768-dim, all-mpnet-base-v2)
    """
    CREATE VECTOR INDEX entity_vector_index IF NOT EXISTS
    FOR (e:Entity) ON (e.embedding)
    OPTIONS {indexConfig: {`vector.dimensions`: 768, `vector.similarity_function`: 'cosine'}}
    """,

    # Vector index on Context embeddings (for optional direct chunk retrieval)
    """
    CREATE VECTOR INDEX context_vector_index IF NOT EXISTS
    FOR (c:Context) ON (c.embedding)
    OPTIONS {indexConfig: {`vector.dimensions`: 768, `vector.similarity_function`: 'cosine'}}
    """,
]


def setup_indexes(driver, db):
    with driver.session(database=db) as session:
        for cypher in SETUP_CYPHER:
            try:
                session.run(cypher)
            except Exception as e:
                # ignore already-exists errors
                if "already exists" not in str(e).lower():
                    print(f"  ⚠️  Schema warning: {e}")
    print("✅  Schema / indexes ready.")


# ── upsert helpers ─────────────────────────────────────────────────────────────

def upsert_context(session, chunk_id, title, text, embedding):
    session.run(
        """
        MERGE (c:Context {chunk_id: $chunk_id})
        SET c.title     = $title,
            c.text      = $text,
            c.embedding = $embedding
        """,
        chunk_id=chunk_id, title=title, text=text, embedding=embedding,
    )


def upsert_entity(session, name, description, embedding):
    session.run(
        """
        MERGE (e:Entity {name: $name})
        SET e.description = $description,
            e.embedding   = $embedding
        """,
        name=name, description=description, embedding=embedding,
    )


def link_entity_to_context(session, entity_name, chunk_id):
    session.run(
        """
        MATCH (e:Entity {name: $name})
        MATCH (c:Context {chunk_id: $chunk_id})
        MERGE (e)-[:MENTIONED_IN]->(c)
        """,
        name=entity_name, chunk_id=chunk_id,
    )


def link_entities_related(session, name_a, name_b, question_id):
    """
    Add a RELATED_TO relationship between two entities that co-appear
    as supporting context for the same multi-hop question.
    """
    session.run(
        """
        MATCH (a:Entity {name: $na})
        MATCH (b:Entity {name: $nb})
        MERGE (a)-[r:RELATED_TO]->(b)
        SET r.via = coalesce(r.via, []) + [$qid]
        """,
        na=name_a, nb=name_b, qid=question_id,
    )


# ── main ingestion ─────────────────────────────────────────────────────────────

def ingest(csv_path: str, db: str, batch_size: int):
    print(f"📂  Loading {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"    {len(df)} rows loaded.")

    print(f"🔧  Loading embedding model ({MODEL_NAME}) ...")
    model = SentenceTransformer(MODEL_NAME)

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()
    print(f"✅  Connected to Neo4j at {NEO4J_URI} — database: {db}")

    setup_indexes(driver, db)

    total_contexts = 0
    total_entities = 0
    total_links    = 0
    total_related  = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Ingesting rows"):
        q_id    = str(row.get("id", ""))
        paragraphs = extract_paragraphs(row.get("context", ""))

        if not paragraphs:
            continue

        # Build list of (title, text, embedding) for this question's context
        texts     = [text  for _, text  in paragraphs]
        titles    = [title for title, _ in paragraphs]
        embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=False)

        entity_names_this_q = []

        with driver.session(database=db) as session:
            for (title, text), emb in zip(paragraphs, embeddings):
                emb_list = emb.tolist()
                chunk_id = f"{q_id}__{title[:60].replace(' ', '_')}"

                # 1. Context node
                upsert_context(session, chunk_id, title, text, emb_list)
                total_contexts += 1

                # 2. Entity node (one per article title; use text as description)
                upsert_entity(session, name=title, description=text[:300], embedding=emb_list)
                total_entities += 1

                # 3. MENTIONED_IN link
                link_entity_to_context(session, title, chunk_id)
                total_links += 1

                entity_names_this_q.append(title)

            # 4. RELATED_TO between all entities in this question's supporting context
            #    This creates the multi-hop bridges GraphRAG can traverse
            for i in range(len(entity_names_this_q)):
                for j in range(i + 1, len(entity_names_this_q)):
                    link_entities_related(
                        session,
                        entity_names_this_q[i],
                        entity_names_this_q[j],
                        q_id,
                    )
                    total_related += 1

    driver.close()

    print(f"""
✅  Ingestion complete!
    Context nodes  : {total_contexts}
    Entity nodes   : {total_entities}
    MENTIONED_IN   : {total_links}
    RELATED_TO     : {total_related}
""")


# ── entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",   default="hotpotqa_sampled_100.csv",
                        help="Path to HotpotQA CSV with context column")
    parser.add_argument("--db",    default=NEO4J_DB,
                        help="Neo4j database name (default: hotpotqarag)")
    parser.add_argument("--batch", type=int, default=32,
                        help="Embedding batch size (default: 32)")
    args = parser.parse_args()

    ingest(args.csv, args.db, args.batch)
