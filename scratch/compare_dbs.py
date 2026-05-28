import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

def check_db(db_name):
    print(f"\n--- Checking Database: {db_name} ---")
    try:
        conn = psycopg2.connect(
            host=DB_HOST, port=DB_PORT,
            user=DB_USER, password=DB_PASSWORD,
            dbname=db_name
        )
        cur = conn.cursor()
        
        cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';")
        tables = [t[0] for t in cur.fetchall()]
        print(f"Tables: {tables}")
        
        if 'context_embeddings' in tables:
            cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'context_embeddings';")
            cols = [c[0] for c in cur.fetchall()]
            print(f"Columns in context_embeddings: {cols}")
            
            cur.execute("SELECT context FROM context_embeddings LIMIT 1;")
            ctx = cur.fetchone()
            if ctx:
                print(f"Sample context (first 100 chars): {ctx[0][:100]}...")
        
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_db("vector_rag")
    check_db("vector_chunk_rag")
