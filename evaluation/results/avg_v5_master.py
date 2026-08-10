import json
from statistics import mean

with open("evaluation/results/v5_20260810_010024.json", encoding="utf-8") as f:
    results = json.load(f)

def avg(values):
    return mean(v for v in values if v is not None)

print("Recall:", avg(r["overall_metrics"]["recall_at_k"] for r in results))
print("Precision:", avg(r["overall_metrics"]["precision_at_k"] for r in results))
print("Hit Rate:", avg(r["overall_metrics"]["hit_rate"] for r in results))
print("MRR:", avg(r["overall_metrics"]["mrr"] for r in results))

print("Correctness:", avg(r["judge_evaluation"]["correctness"]["score"] for r in results))
print("Groundedness:", avg(r["judge_evaluation"]["groundedness"]["score"] for r in results))
print("Completeness:", avg(r["judge_evaluation"]["completeness"]["score"] for r in results))
print("Hallucination:", avg(r["judge_evaluation"]["hallucination"]["score"] for r in results))

print("Latency:", avg(r["overall_metrics"]["total_latency"] for r in results))
print("Cost:", avg(r["overall_metrics"]["total_cost"] for r in results))
print("Total cost:", sum(r["overall_metrics"]["total_cost"] for r in results))