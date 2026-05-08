import sys
import io

# Ensure Windows prints emojis and complex characters without crashing
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from vector_rag_search import VectorRAGSearch

def main():
    print("Initializing VectorRAGSearch...")
    searcher = VectorRAGSearch()
    
    # You can change this query to anything you want to test!
    query = "What is in front of the Notre Dame Main Building?"
    print(f"\nSearching for: '{query}'\n")
    
    # Find the 5 most relevant embeddings using cosine similarity
    results = searcher.search(query, top_k=5)
    
    print(f"Found {len(results)} results:\n" + "="*50)
    
    for i, r in enumerate(results):
        print(f"Result #{i+1}")
        print(f"Title: {r['title']}")
        print(f"Similarity Score: {r['similarity']}")
        print("Context Snippet:")
        # Truncating to 300 characters just to keep the terminal readable, 
        # remove the [:300] if you want to see the whole giant chunk of text
        print(r['context'][:300] + "...") 
        print("-" * 50)
        
    searcher.close()

if __name__ == "__main__":
    main()
