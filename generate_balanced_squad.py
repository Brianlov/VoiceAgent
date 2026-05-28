import pandas as pd
import numpy as np
import os
import re

BUILTIN_SAMPLES = [
    {"question": "To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?",
     "ground_truth": "Saint Bernadette Soubirous"},
    {"question": "The granting of Doctorate degrees first occurred in what year at Notre Dame?",
     "ground_truth": "1924"},
    {"question": "When did the first art gallery open in Washington state?",
     "ground_truth": "1927"},
    {"question": "What can cause your memory to deterioriate or not work as well?",
     "ground_truth": "Stress"},
    {"question": "What has excessive hunting contributed heavily to?",
     "ground_truth": "the endangerment, extirpation and extinction of many animals"},
    {"question": "What is the name of the AFL team based in Tampa Bay?",
     "ground_truth": "Storm"},
    {"question": "Which conjugation has about 3500 verbs?",
     "ground_truth": "first conjugation"},
    {"question": "Which university library is larger than Nanjing University Library?",
     "ground_truth": "Peking University Library"},
    {"question": "Which canton is Berne the capital?",
     "ground_truth": "Canton of Bern"},
    {"question": "Why is it difficult to measure corruption?",
     "ground_truth": "imprecise definitions of corruption"},
    {"question": "Why is Yiddish not a dialect of German?",
     "ground_truth": "a Yiddish speaker would not consult a German dictionary"},
    # --- MULTI-HOP REASONING SAMPLES ---
    {"question": "Who is the patron saint of the university where the Scholastic Magazine began publishing in 1876?",
     "ground_truth": "The Virgin Mary"},
    {"question": "What hormones damage the brain region associated with memory loss, and what daily activity helps stabilize those memories?",
     "ground_truth": "Glucocorticoids damage the hippocampal region, and sleep helps stabilize memories."},
    {"question": "How many years after the Scholastic Magazine began publishing did Notre Dame first formally offer Doctorate degrees?",
     "ground_truth": "48 years"},
    {"question": "When regals recruited low-ranking local tribes for expeditions in British India, what specific global consequence did this type of activity heavily contribute to?",
     "ground_truth": "The endangerment, extirpation and extinction of many animals."},
    {"question": "In the city where the Henry Art Gallery opened as a public art museum, what type of transportation system did they later begin to focus on?",
     "ground_truth": "Mass transit."}
]

def main():
    csv_path = "squad_train_dataset.csv"
    output_path = "squad_balanced_500.csv"
    
    if not os.path.exists(csv_path):
        print(f"Error: Cannot find {csv_path} in the current directory.")
        return

    print(f"Reading SQuAD dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} total rows from original dataset.")

    # Clean the dataset
    df = df.dropna(subset=["question", "answers"])

    # Classify the original dataset into Reasoning vs Factual
    pattern = re.compile(r'\b(why|when)\b', re.IGNORECASE)
    is_reasoning = df["question"].apply(lambda q: bool(pattern.search(str(q))))
    original_reasoning_df = df[is_reasoning]
    original_factual_df = df[~is_reasoning]

    # Classify the built-in samples into Reasoning vs Factual
    builtin_reasoning = []
    builtin_factual = []
    for item in BUILTIN_SAMPLES:
        if pattern.search(item["question"]):
            builtin_reasoning.append(item)
        else:
            # Let's also treat the multi-hop reasoning questions as reasoning
            # since they are clearly reasoning/multi-hop questions
            if "Who is the patron" in item["question"] or "What hormones" in item["question"] or "How many years" in item["question"] or "In the city" in item["question"]:
                builtin_reasoning.append(item)
            else:
                builtin_factual.append(item)

    print(f"Built-in split: {len(builtin_reasoning)} Reasoning, {len(builtin_factual)} Factual.")

    # Target is exactly 250 Reasoning and 250 Factual (Total 500)
    target_reasoning = 250
    target_factual = 250

    needed_reasoning = target_reasoning - len(builtin_reasoning)
    needed_factual = target_factual - len(builtin_factual)

    print(f"Sampling additional {needed_reasoning} Reasoning and {needed_factual} Factual questions from CSV...")

    sampled_reasoning = original_reasoning_df.sample(n=needed_reasoning, random_state=42)
    sampled_factual = original_factual_df.sample(n=needed_factual, random_state=42)

    # Convert built-ins to dataframes matching the columns
    builtin_rows = []
    for i, s in enumerate(BUILTIN_SAMPLES):
        builtin_rows.append({
            "id": f"builtin_{i}",
            "title": "builtin",
            "context": "",
            "question": s["question"],
            "answers": s["ground_truth"]
        })
    builtin_df = pd.DataFrame(builtin_rows)

    # Combine all components
    final_df = pd.concat([builtin_df, sampled_reasoning, sampled_factual], ignore_index=True)
    
    # Shuffle only the non-builtin entries OR shuffle everything but make sure they are well-mixed
    # Let's prepend the 16 built-ins first so they are guaranteed to be in any smaller sub-sample runs!
    # That way if someone runs --n-samples 20, they get the 16 built-ins + 4 random ones.
    # So we keep the 16 built-ins at the top of the file, and shuffle the rest!
    shuffled_sampled = pd.concat([sampled_reasoning, sampled_factual], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
    balanced_df = pd.concat([builtin_df, shuffled_sampled], ignore_index=True)

    # Save to CSV
    balanced_df.to_csv(output_path, index=False)
    print(f"Success! Saved balanced dataset to: {output_path}")
    print(f"Summary of {output_path}:")
    print(f"   - Total Samples: {len(balanced_df)}")
    print(f"   - Built-in Samples at the top: {len(builtin_df)}")
    print(f"   - Reasoning Samples total: {target_reasoning}")
    print(f"   - Factual Samples total: {target_factual}")

if __name__ == "__main__":
    main()
