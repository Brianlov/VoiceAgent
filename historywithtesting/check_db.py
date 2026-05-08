import sys
import io
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from vector_rag_search import VectorRAGSearch

searcher = VectorRAGSearch()
query = "To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?"
results = searcher.search(query, 5)

for i, r in enumerate(results):
    print(f"\nResult {i+1}: Title: {r['title']}, Similarity: {r['similarity']}")
    print(r['context'][:150])
