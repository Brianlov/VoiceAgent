import pandas as pd
import json
import ast
import sys

def main():
    csv_path = "results/checkpoint_graph_raw.csv"
    df = pd.read_csv(csv_path)
    
    failed_rows = []
    
    for idx, row in df.iterrows():
        ans = str(row["answer"])
        gt = str(row["ground_truth"])
        q = str(row["question"])
        
        # Check if the response is "I'm sorry..."
        if "couldn't find" in ans or "sorry" in ans:
            # Parse contexts list
            contexts_str = row["contexts"]
            try:
                contexts_list = ast.literal_eval(contexts_str)
            except Exception:
                try:
                    contexts_list = json.loads(contexts_str.replace("'", '"'))
                except Exception:
                    contexts_list = [contexts_str]
            
            # Combine all contexts text into one lowercased string
            combined_context = " ".join(contexts_list).lower()
            
            # Check if ground_truth is in combined context
            if gt.lower() in combined_context:
                failed_rows.append({
                    "csv_row": idx + 2, # 1-based index plus header
                    "index": idx,
                    "question": q,
                    "ground_truth": gt,
                    "contexts": contexts_list,
                    "answer": ans
                })

    # Write full analysis to a text file using UTF-8 to avoid console encoding crashes
    output_path = "scratch/failed_analysis.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"Total failed rows where ground_truth is ACTUALLY present in retrieved context: {len(failed_rows)}\n")
        for item in failed_rows:
            f.write("\n======================================================================\n")
            f.write(f"Row {item['csv_row']}: Q: {item['question']}\n")
            f.write(f"Ground Truth: '{item['ground_truth']}'\n")
            f.write(f"Answer Given: '{item['answer']}'\n")
            f.write("Retrieved Context Chunks:\n")
            for c in item["contexts"]:
                if item["ground_truth"].lower() in c.lower():
                    f.write(f"  * [CONTAINED GT] -> {c}\n")
                else:
                    f.write(f"    - [OTHER CHUNK] -> {c}\n")

    print(f"Successfully wrote {len(failed_rows)} analyzed cases to scratch/failed_analysis.txt")

if __name__ == "__main__":
    main()
