import json
data = json.load(open('hotpotqa_test_100.json', encoding='utf-8'))
indices = [3, 8, 10, 15, 25, 28, 34, 39, 45, 46, 49, 50, 51, 52, 62, 67, 72, 74, 78, 82, 98, 99]
for i in indices:
    q = data[i]
    print(f"[idx {i} / CSV line {i+2}]")
    print(f"  Q: {q['question']}")
    print(f"  A: {q['answers']}")
    print()
