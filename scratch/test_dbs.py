import os
import psycopg2
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

def test_postgres():
    print("Testing Postgres connection...")
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    dbname = os.getenv("DB_NAME", "vector_chunk_rag2000")
    user = os.getenv("DB_USER", "postgres")
    password = os.getenv("DB_PASSWORD", "20010808")
    
    try:
        conn = psycopg2.connect(host=host, port=port, dbname=dbname, user=user, password=password)
        print(f"[OK] Successfully connected to Postgres: {dbname}")
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM context_embeddings;")
        count = cur.fetchone()[0]
        print(f"📊 Rows in context_embeddings: {count}")
        cur.close()
        conn.close()
    except Exception as e:
        print(f"[ERROR] Postgres connection failed: {e}")

def test_neo4j():
    print("\nTesting Neo4j connection...")
    uri = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "20010808")
    db = os.getenv("NEO4J_DB", "llmknowledgegraph")
    
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        print(f"[OK] Successfully connected to Neo4j at {uri}")
        
        with driver.session(database=db) as session:
            result = session.run("MATCH (n) RETURN count(n) AS c;")
            count = result.single()["c"]
            print(f"📊 Nodes in database '{db}': {count}")
            
            result_label = session.run("MATCH (n) RETURN labels(n) AS l, count(*) AS c;")
            for record in result_label:
                print(f"   Label {record['l']}: {record['c']}")
                
        driver.close()
    except Exception as e:
        print(f"[ERROR] Neo4j connection failed: {e}")

if __name__ == "__main__":
    test_postgres()
    test_neo4j()
