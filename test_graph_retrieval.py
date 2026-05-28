"""
test_graph_retrieval.py
=======================
Tests GraphRAG retrieval on specific question indices from hotpotqa_test_100.json.
Prints whether context was found and a snippet.
"""

import asyncio
import json
import os
from dotenv import load_dotenv
from Graph_service_purever4 import GraphRAGProcessorV4 as GraphRAGProcessor

load_dotenv()

# CSV line numbers (header = line 1), converted to 0-based JSON indices: line - 2
# Original CSV lines: 5 10 12 17 27 30 36 41 47 48 51 52 53 54 64 69 74 76 80 84 100 101
TARGET_INDICES = [3, 8, 10, 15, 25, 28, 34, 39, 45, 46, 49, 50, 51, 52, 62, 67, 72, 74, 78, 82, 98, 99]

with open("hotpotqa_test_100.json", encoding="utf-8") as f:
    all_questions = json.load(f)

# Select questions at those 0-based indices
selected = []
for i in TARGET_INDICES:
    if 0 <= i < len(all_questions):
        selected.append((i + 2, all_questions[i]))  # show original CSV line = i+2
    else:
        print(f"  [0-based idx {i}] OUT OF RANGE (file has {len(all_questions)} questions)")


async def main():
    processor = GraphRAGProcessor(
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687"),
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", "20010808"),
        neo4j_db=os.getenv("NEO4J_DB", "hotpotqarag"),
    )
    await processor._initialize_graph()

    found = 0
    not_found = 0

    for idx, item in selected:
        q  = item["question"]
        gt = item["answers"]
        print(f"\n{'='*70}")
        print(f"[Q{idx}] {q}")
        print(f"  Expected: {gt}")

        ctx = await processor.retrieve(q, as_list=True)

        if ctx:
            found += 1
            print(f"  Context chunks retrieved: {len(ctx)}")
            # Show first 2 chunks truncated
            for i, chunk in enumerate(ctx[:2]):
                snippet = str(chunk)[:150].replace("\n", " ")
                print(f"  [{i+1}] {snippet}...")
        else:
            not_found += 1
            print(f"  !! NO CONTEXT FOUND")

    await processor.driver.close()

    print(f"\n{'='*70}")
    print(f"SUMMARY: {found} found / {not_found} not found  (out of {len(selected)} questions)")


asyncio.run(main())
