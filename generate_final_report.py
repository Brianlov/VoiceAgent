import pandas as pd
from pathlib import Path
from ragas_benchmark import compare_results, export_results, plot_all, RENAME

import glob
import os

# Define final output run directory
final_run_dir = Path("results_hotpotqa/final_comparison")
final_run_dir.mkdir(parents=True, exist_ok=True)

# Path to the completed samples CSV files
vector_files = glob.glob("results_hotpotqa/*/vectorrag_per_query_eval.csv")
graph_files = glob.glob("results_hotpotqa/*/graphrag_per_query_eval.csv")

if not vector_files:
    print("Error: VectorRAG scores file not found.")
    exit(1)

if not graph_files:
    print("Error: GraphRAG scores file not found.")
    exit(1)

vector_scores_file = Path(max(vector_files, key=os.path.getctime))
graph_scores_file = Path(max(graph_files, key=os.path.getctime))

# Load completed DataFrames
print("Loading VectorRAG and GraphRAG 300-sample results...")
vec_df = pd.read_csv(vector_scores_file)
graph_df = pd.read_csv(graph_scores_file)

# Compute mean aggregates for RAGAS score columns
score_cols = [v for v in RENAME.values() if v in vec_df.columns]
vec_agg = {col: float(vec_df[col].mean()) for col in score_cols}
graph_agg = {col: float(graph_df[col].mean()) for col in score_cols}

print("\nComputed VectorRAG Averages:")
for k, v in vec_agg.items():
    print(f"  • {k:<20} {v:.4f}")

print("\nComputed GraphRAG Averages:")
for k, v in graph_agg.items():
    print(f"  • {k:<20} {v:.4f}")

# Generate the comparison table
cmp_df = compare_results(vec_agg, graph_agg)

# Export the combined CSV results and summary
export_results(vec_df, graph_df, cmp_df, final_run_dir)

# Generate and save all 4 premium charts
plot_all(vec_df, graph_df, vec_agg, graph_agg, final_run_dir)

print(f"\n🎉 SUCCESS: All charts and reports have been generated and exported to: {final_run_dir.resolve()}\n")
