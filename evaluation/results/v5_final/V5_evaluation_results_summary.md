# Developer Docs RAG — V5 Evaluation Results

## Benchmark Overview

| Category | Questions |
|---|---:|
| General | 52 |
| Code Review | 20 |
| Debugging | 10 |
| **Total** | **82** |

## Overall V5 Baseline

Weighted across all 82 questions:

| Metric | Overall |
|---|---:|
| Recall@K | **0.6241** |
| Precision@K | **0.3957** |
| Hit Rate | **0.9024** |
| MRR | **0.7430** |
| Correctness | **4.756 / 5** |
| Groundedness | **4.793 / 5** |
| Completeness | **4.756 / 5** |
| Hallucination | **0.305 / 5** |
| Average latency | **28.97 s** |
| Average cost/question | **$0.002139** |
| Average tokens/question | **3472** |
| Total benchmark cost | **$0.17540980** |

## Category Results

| Metric | General (52) | Code Review (20) | Debugging (10) |
|---|---:|---:|---:|
| Recall@K | 0.6224 | 0.6021 | 0.6767 |
| Precision@K | 0.4374 | 0.3027 | 0.3646 |
| Hit Rate | 0.9038 | 0.8500 | 1.0000 |
| MRR | 0.8120 | 0.5476 | 0.7750 |
| Correctness | 4.846 | 4.550 | 4.700 |
| Groundedness | 4.846 | 4.650 | 4.800 |
| Completeness | 4.846 | 4.550 | 4.700 |
| Hallucination | 0.212 | 0.550 | 0.300 |
| Avg latency | 27.21s | 31.55s | 33.00s |
| Avg cost/question | $0.002000 | $0.002507 | $0.002127 |
| Total cost | $0.1040004 | $0.05013615 | $0.02127325 |

## Investigation Candidates

These are candidates for manual investigation, **not confirmed errors**.

### General — 10
Q47, Q40, Q41, Q48, Q42, Q43, Q46, Q49, Q50, Q11

### Code Review — 10
Q16, Q9, Q20, Q10, Q11, Q5, Q15, Q2, Q12, Q19

### Debugging — 3
Q2, Q7, Q8

**Total: 23 category-specific question entries.**

The same numeric ID in different categories represents a different question.

## Key Findings

1. Final-answer quality is strong across all three categories.
2. Debugging has the strongest retrieval profile: Recall 0.6767 and Hit Rate 1.0.
3. Code Review is the weakest retrieval category, especially MRR 0.5476 and Precision 0.3027.
4. General has the strongest MRR at 0.8120.
5. Low Recall does not always correspond to a poor final answer.
6. Some `expected_files` may be broader than the evidence strictly required to answer a question.
7. Code Review has the highest hallucination average at 0.55/5, largely around architectural inferences not explicitly supported by retrieved context.
8. The Judge is useful as a triage/evaluation mechanism, but is not absolute ground truth.
9. Do not tune the RAG system simply to improve these individual cases before freezing the baseline.

## Recommended Next Step

```text
V5 — 82-question baseline
        ↓
Optionally audit the 23 candidate questions
        ↓
Freeze dataset + baseline
        ↓
Move to Agentic RAG
        ↓
Run the same 82 questions
        ↓
Compare V5 vs Agentic RAG
```

Primary comparison metrics:

- Recall@K
- Precision@K
- Hit Rate
- MRR
- Correctness
- Groundedness
- Completeness
- Hallucination
- Average end-to-end latency
- Average cost/question
- Total benchmark cost

> **Important:** The overall values above are question-count-weighted averages of the three reported category averages. For the most rigorous final report, recomputing from all 82 individual result records is preferable if needed.
