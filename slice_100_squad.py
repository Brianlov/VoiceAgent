import pandas as pd
from pathlib import Path
from ragas_benchmark import compare_results, export_results, plot_all, RENAME

# Define output directory specifically as requested
output_dir = Path("results_ollama_100")
output_dir.mkdir(parents=True, exist_ok=True)

# Path to the completed 300 samples CSV files
vector_scores_file = Path("results_ollama/checkpoint_vector_scores_ollama.csv")
graph_scores_file = Path("results_ollama/20260527_122832/graphrag_per_query_eval.csv")

if not vector_scores_file.exists():
    print(f"Error: VectorRAG scores file not found at {vector_scores_file}")
    exit(1)

if not graph_scores_file.exists():
    print(f"Error: GraphRAG scores file not found at {graph_scores_file}")
    exit(1)

# Load full DataFrames
print("Loading 300-sample completed evaluation datasets...")
vec_df_full = pd.read_csv(vector_scores_file)
graph_df_full = pd.read_csv(graph_scores_file)

# Slice down to exactly the first 100 samples
print(f"Slicing datasets to exactly the first 100 rows...")
vec_df_100 = vec_df_full.head(100).copy()
graph_df_100 = graph_df_full.head(100).copy()

# Compute mean aggregates for the 100-sample subset
score_cols = [v for v in RENAME.values() if v in vec_df_100.columns]
vec_agg = {col: float(vec_df_100[col].mean()) for col in score_cols}
graph_agg = {col: float(graph_df_100[col].mean()) for col in score_cols}

print("\nComputed VectorRAG (100-sample subset) Averages:")
for k, v in vec_agg.items():
    print(f"  • {k:<20} {v:.4f}")

print("\nComputed GraphRAG (100-sample subset) Averages:")
for k, v in graph_agg.items():
    print(f"  • {k:<20} {v:.4f}")

# Generate the comparison table for the 100-sample subset
cmp_df = compare_results(vec_agg, graph_agg)

# Export the 100-sample CSV files and summary comparison to results_ollama_100
export_results(vec_df_100, graph_df_100, cmp_df, output_dir)

# Generate and save all 4 premium charts inside results_ollama_100
plot_all(vec_df_100, graph_df_100, vec_agg, graph_agg, output_dir)

print(f"\n🎉 SUCCESS: Option B complete! All 100-sample charts and reports saved to: {output_dir.resolve()}\n")
