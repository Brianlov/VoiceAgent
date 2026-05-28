import pandas as pd
import glob
import os

def compare_evals():
    # Find latest eval files
    vector_files = glob.glob("results_hotpotqa/*/vectorrag_per_query_eval.csv")
    graph_files = glob.glob("results_hotpotqa/*/graphrag_per_query_eval.csv")
    
    if not vector_files or not graph_files:
        print("Missing evaluation files.")
        return
        
    latest_vector = max(vector_files, key=os.path.getctime)
    latest_graph = max(graph_files, key=os.path.getctime)
    
    df_v = pd.read_csv(latest_vector)
    df_g = pd.read_csv(latest_graph)
    
    metrics = ["Faithfulness", "AnswerRelevancy", "ContextPrecision", "ContextRecall", "answer_correctness"]
    
    print(f"VectorRAG File: {latest_vector} ({len(df_v)} samples)")
    print(f"GraphRAG File : {latest_graph} ({len(df_g)} samples)")
    print("-" * 50)
    print(f"{'Metric':<25} | {'VectorRAG':<10} | {'GraphRAG':<10} | {'Diff'}")
    print("-" * 50)
    
    for m in metrics:
        v_mean = df_v[m].mean() if m in df_v.columns else 0.0
        g_mean = df_g[m].mean() if m in df_g.columns else 0.0
        diff = g_mean - v_mean
        
        sign = "+" if diff > 0 else ""
        print(f"{m:<25} | {v_mean:.4f}     | {g_mean:.4f}     | {sign}{diff:.4f}")

    print("-" * 50)
    
    # Also calculate retrieval/generation time averages
    print("\nLatency (ms):")
    v_ret = df_v["retrieval_ms"].mean() if "retrieval_ms" in df_v.columns else 0
    v_gen = df_v["generation_ms"].mean() if "generation_ms" in df_v.columns else 0
    g_ret = df_g["retrieval_ms"].mean() if "retrieval_ms" in df_g.columns else 0
    g_gen = df_g["generation_ms"].mean() if "generation_ms" in df_g.columns else 0
    
    print(f"{'Retrieval (ms)':<25} | {v_ret:.1f}      | {g_ret:.1f}")
    print(f"{'Generation (ms)':<25} | {v_gen:.1f}      | {g_gen:.1f}")
    
if __name__ == "__main__":
    compare_evals()
