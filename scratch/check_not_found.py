import pandas as pd

def main():
    vec_df = pd.read_csv("results/checkpoint_vector_raw.csv")
    graph_df = pd.read_csv("results/checkpoint_graph_raw.csv")
    
    vec_not_found = vec_df["answer"].str.contains("couldn't find|sorry", case=False, na=True).sum()
    graph_not_found = graph_df["answer"].str.contains("couldn't find|sorry", case=False, na=True).sum()
    
    print(f"Total evaluated samples: {len(vec_df)}")
    print(f"VectorRAG 'not found' count: {vec_not_found}  ({(vec_not_found/len(vec_df))*100:.1f}%)")
    print(f"GraphRAG  'not found' count: {graph_not_found}  ({(graph_not_found/len(graph_df))*100:.1f}%)")

if __name__ == "__main__":
    main()
