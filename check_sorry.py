import pandas as pd

vec = pd.read_csv(r'generations_hotpotqa\vectorrag_generations_ollama.csv')
graph = pd.read_csv(r'generations_hotpotqa\graphrag_generations_ollama.csv')

vec_sorry = vec['answer'].str.contains("I'm sorry", na=False).sum()
graph_sorry = graph['answer'].str.contains("I'm sorry", na=False).sum()

print(f'VectorRAG sorry: {vec_sorry} / {len(vec)}')
print(f'GraphRAG sorry:  {graph_sorry} / {len(graph)}')
