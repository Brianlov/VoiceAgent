# vector_rag_search.py

import psycopg2
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "vector_chunk_rag")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

MODEL_NAME = "all-mpnet-base-v2"

class VectorRAGSearch:
    def __init__(self):
        print(f"🔧 Loading model: {MODEL_NAME}...")
        self.model = SentenceTransformer(MODEL_NAME)
        self.conn = psycopg2.connect(
            host=DB_HOST, port=DB_PORT,
            user=DB_USER, password=DB_PASSWORD,
            dbname=DB_NAME
        )
        print(f"✅ Connected to database: {DB_NAME}")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Semantic search using cosine distance (<=>).
        Returns top_k most similar contexts.
        """
        # Embed the query
        query_embedding = self.model.encode(query).tolist()
        emb_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT
                chunk_id,
                context_idx,
                title,
                context,
                1 - (embedding <=> %s::vector) AS similarity
            FROM context_embeddings
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (emb_str, emb_str, top_k)
        )

        results = []
        for row in cur.fetchall():
            results.append({
                "chunk_id": row[0],
                "context_idx": row[1],
                "title": row[2],
                "context": row[3],
                "similarity": round(float(row[4]), 4)
            })
        cur.close()
        return results

    def close(self):
        if self.conn:
            self.conn.close()
