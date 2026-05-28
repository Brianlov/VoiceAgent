import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "vector_chunk_rag")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

def check_schema():
    try:
        conn = psycopg2.connect(
            host=DB_HOST, port=DB_PORT,
            user=DB_USER, password=DB_PASSWORD,
            dbname=DB_NAME
        )
        cur = conn.cursor()
        
        print("--- Tables ---")
        cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';")
        for table in cur.fetchall():
            print(table[0])
            
        print("\n--- Columns in context_embeddings ---")
        cur.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'context_embeddings';")
        for col in cur.fetchall():
            print(f"{col[0]}: {col[1]}")
            
        print("\n--- Sample row from context_embeddings ---")
        cur.execute("SELECT * FROM context_embeddings LIMIT 1;")
        row = cur.fetchone()
        if row:
            colnames = [desc[0] for desc in cur.description]
            for i, val in enumerate(row):
                v_str = str(val)[:100] + "..." if isinstance(val, str) and len(val) > 100 else val
                print(f"{colnames[i]}: {v_str}")
        
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_schema()
