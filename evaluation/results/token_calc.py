import json
from pathlib import Path

files = [
    Path("evaluation/results/v5_20260810_010024.json"),
    Path("evaluation/results/v5_20260809_004318.json"),
    Path("evaluation/results/v5_20260808_212209.json"),
]

results = []

for file in files:
    with file.open("r", encoding="utf-8") as f:
        results.extend(json.load(f))

avg_input = sum(r["generation_metrics"]["prompt_tokens"] for r in results) / len(results)
avg_output = sum(r["generation_metrics"]["completion_tokens"] for r in results) / len(results)
avg_total = sum(r["generation_metrics"]["total_tokens"] for r in results) / len(results)

print(f"Questions: {len(results)}")
print(f"Average input tokens: {avg_input:.2f}")
print(f"Average output tokens: {avg_output:.2f}")
print(f"Average total tokens: {avg_total:.2f}")