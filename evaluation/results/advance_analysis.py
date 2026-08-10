import json
from collections import Counter
from pathlib import Path

RESULT_PATH = Path("evaluation/results/v5_20260810_010024.json")

with RESULT_PATH.open(encoding="utf-8") as f:
    results = json.load(f)


# ---------------------------------------------------------
# 1. Score distributions
# ---------------------------------------------------------

def distribution(values):
    return dict(sorted(Counter(values).items()))


print("\n" + "=" * 70)
print("JUDGE SCORE DISTRIBUTIONS")
print("=" * 70)

for metric in ["correctness", "groundedness", "completeness", "hallucination"]:
    values = [
        r["judge_evaluation"][metric]["score"]
        for r in results
    ]

    print(f"\n{metric.capitalize()}:")
    print(distribution(values))


# ---------------------------------------------------------
# 2. Retrieval metric distributions
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("RETRIEVAL METRIC DISTRIBUTIONS")
print("=" * 70)

for metric in ["recall_at_k", "precision_at_k", "hit_rate", "mrr"]:
    values = [
        r["overall_metrics"][metric]
        for r in results
    ]

    print(f"\n{metric}:")
    print(distribution(values))


# ---------------------------------------------------------
# 3. Worst retrieval questions
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("LOWEST RECALL")
print("=" * 70)

for r in sorted(
    results,
    key=lambda x: x["overall_metrics"]["recall_at_k"]
)[:10]:

    print(
        f'Q{r["question"]["id"]} | '
        f'Recall={r["overall_metrics"]["recall_at_k"]:.3f} | '
        f'Hit={r["overall_metrics"]["hit_rate"]:.1f} | '
        f'MRR={r["overall_metrics"]["mrr"]:.3f}'
    )
    print(f'  {r["question"]["question"]}')


# ---------------------------------------------------------
# 4. Worst Judge questions
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("LOWEST JUDGE SCORES")
print("=" * 70)

def judge_average(r):
    j = r["judge_evaluation"]

    return (
        j["correctness"]["score"]
        + j["groundedness"]["score"]
        + j["completeness"]["score"]
    ) / 3


for r in sorted(results, key=judge_average)[:10]:

    j = r["judge_evaluation"]

    print(
        f'Q{r["question"]["id"]} | '
        f'Correct={j["correctness"]["score"]} | '
        f'Ground={j["groundedness"]["score"]} | '
        f'Complete={j["completeness"]["score"]} | '
        f'Hallucination={j["hallucination"]["score"]}'
    )
    print(f'  {r["question"]["question"]}')


# ---------------------------------------------------------
# 5. Hallucination cases
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("HALLUCINATION > 0")
print("=" * 70)

hallucination_cases = [
    r for r in results
    if r["judge_evaluation"]["hallucination"]["score"] > 0
]

for r in hallucination_cases:

    j = r["judge_evaluation"]

    print(
        f'\nQ{r["question"]["id"]} | '
        f'Hallucination={j["hallucination"]["score"]}'
    )

    print("Unsupported claims:")
    for claim in j.get("unsupported_claims", []):
        print(f"  - {claim}")

    print("Justification:")
    print(f'  {j["hallucination"]["justification"]}')


# ---------------------------------------------------------
# 6. Low retrieval but good answer
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("LOW RECALL + GOOD ANSWER")
print("=" * 70)

cases = []

for r in results:

    recall = r["overall_metrics"]["recall_at_k"]
    judge_avg = judge_average(r)

    if recall < 0.5 and judge_avg >= 4.0:
        cases.append((r, recall, judge_avg))

for r, recall, judge_avg in sorted(cases, key=lambda x: x[1]):

    print(
        f'\nQ{r["question"]["id"]} | '
        f'Recall={recall:.3f} | '
        f'JudgeAvg={judge_avg:.2f}'
    )

    print(f'  {r["question"]["question"]}')

    print("  Expected files:")
    for f in r["question"]["expected_files"]:
        print(f"    - {f}")


# ---------------------------------------------------------
# 7. Good retrieval but poor answer
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("GOOD RECALL + POOR ANSWER")
print("=" * 70)

cases = []

for r in results:

    recall = r["overall_metrics"]["recall_at_k"]
    judge_avg = judge_average(r)

    if recall >= 0.75 and judge_avg < 4.0:
        cases.append((r, recall, judge_avg))

for r, recall, judge_avg in sorted(cases, key=lambda x: x[2]):

    print(
        f'\nQ{r["question"]["id"]} | '
        f'Recall={recall:.3f} | '
        f'JudgeAvg={judge_avg:.2f}'
    )

    print(f'  {r["question"]["question"]}')


# ---------------------------------------------------------
# 8. Highest latency
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("HIGHEST LATENCY")
print("=" * 70)

for r in sorted(
    results,
    key=lambda x: x["overall_metrics"]["total_latency"],
    reverse=True
)[:10]:

    print(
        f'Q{r["question"]["id"]} | '
        f'{r["overall_metrics"]["total_latency"]:.2f}s | '
        f'${r["overall_metrics"]["total_cost"]:.6f}'
    )


# ---------------------------------------------------------
# 9. Highest cost
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("HIGHEST COST")
print("=" * 70)

for r in sorted(
    results,
    key=lambda x: x["overall_metrics"]["total_cost"],
    reverse=True
)[:10]:

    print(
        f'Q{r["question"]["id"]} | '
        f'${r["overall_metrics"]["total_cost"]:.6f} | '
        f'{r["overall_metrics"]["total_latency"]:.2f}s'
    )


# ---------------------------------------------------------
# 10. Missing concepts frequency
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("MISSING CONCEPT FREQUENCY")
print("=" * 70)

missing_concepts = Counter()

for r in results:
    missing_concepts.update(
        r["judge_evaluation"].get("missing_concepts", [])
    )

for concept, count in missing_concepts.most_common():
    print(f"{count:>3} | {concept}")


print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)