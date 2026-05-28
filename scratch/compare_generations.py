import pandas as pd
import glob
import os

def check_generations():
    # Find the latest generation files
    graph_files = glob.glob("generations_hotpotqa/graphrag_generations_*.csv")
    vector_files = glob.glob("generations_hotpotqa/vectorrag_generations_*.csv")
    
    if not graph_files:
        print("No graph generations found.")
        return
    if not vector_files:
        print("No vector generations found.")
        return
        
    latest_graph = max(graph_files, key=os.path.getctime)
    latest_vector = max(vector_files, key=os.path.getctime)
    
    df_graph = pd.read_csv(latest_graph)
    df_vector = pd.read_csv(latest_vector)
    
    graph_sorry = df_graph['answer'].str.contains("I'm sorry", na=False, case=False).sum()
    vector_sorry = df_vector['answer'].str.contains("I'm sorry", na=False, case=False).sum()
    
    print(f"--- I'M SORRY COUNT ---")
    print(f"VectorRAG : {vector_sorry} out of {len(df_vector)}")
    print(f"GraphRAG  : {graph_sorry} out of {len(df_graph)}")
    print(f"-----------------------\n")
    
    # Let's inspect some of the hard indices we manually injected
    indices = [3, 8, 10, 15, 25, 28, 34, 39, 45, 46, 49, 50, 51, 52, 62, 67, 72, 74, 78, 82, 98, 99]
    print("--- HARD QUESTIONS COMPARISON ---")
    for idx in indices:
        if idx < len(df_graph) and idx < len(df_vector):
            q = df_graph.iloc[idx]['question']
            g_ans = df_graph.iloc[idx]['answer']
            v_ans = df_vector.iloc[idx]['answer']
            gt_ans = df_graph.iloc[idx]['ground_truth']
            
            print(f"Q: {q}")
            print(f"  GT    : {gt_ans}")
            print(f"  Vector: {v_ans}")
            print(f"  Graph : {g_ans}")
            print()

if __name__ == "__main__":
    check_generations()
