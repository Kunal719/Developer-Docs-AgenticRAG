# Developer Docs Agentic RAG — Evaluation Results

This document records evaluation history from the V5 baseline through V6 retrieval diagnostics and the later Agentic RAG evaluation. These experiments use different datasets and configurations and should not be treated as one controlled leaderboard.

## Metric definitions and interpretation

- **Recall — RRF results:** recall is calculated on the results combined by Reciprocal Rank Fusion (RRF).
- **Precision — reranked selections:** precision is calculated on the documents selected after reranking. It is not precision on the raw RRF candidate pool.
- **Hit Rate:** whether at least one expected relevant item was retrieved.
- **MRR:** rewards relevant evidence appearing earlier in the ranking.
- **Correctness, Groundedness, Completeness, Hallucination:** answer-quality ratings on a 1–5 scale in the source experiments. Lower hallucination is better.
- Latency and cost depend on model configuration, retrieval/generation calls, hardware, and runtime.

Recall and precision refer to different stages of the pipeline. They should be labelled explicitly and not treated as though they were calculated on the same set.

---

## 1. V5 baseline — 82-question benchmark

Evaluated on **82 questions**: 52 general developer questions, 20 code-review questions, and 10 debugging questions.

| Metric | Result |
|---|---:|
| Recall — RRF results | 0.624 |
| Precision — reranked selections (reported at 8) | 0.396 |
| Hit Rate@8 | 0.902 |
| MRR | 0.743 |
| Correctness | 4.756 / 5 |
| Groundedness | 4.793 / 5 |
| Completeness | 4.756 / 5 |
| Hallucination | 0.305 / 5 |
| Average end-to-end latency | 28.97 s |
| Average tokens per question | 3,472 |
| Average cost per question | approximately $0.00214 |

Latency was measured on a CPU-only development machine.

**Interpretation:** the baseline had strong judged answer quality and high hit rate, but retrieval did not always provide all evidence needed for broad, multi-concept developer questions. These are historical V5 results, not measurements of the later Agentic RAG pipeline.

---

## 2. V6 diagnostic experiments — investigating difficult questions

The project investigated failure cases before adding more architecture. The attached `results.md` records a retrieval/agentic-capability study. The cases below illustrate why query strategies such as combined-query and multi-query retrieval were explored.

### Case 1 — Command routing

The question combined `Command(goto=...)`, static `add_edge()` behavior, and checkpoint/resume semantics.

- Original V6: correctness 4/5, groundedness 4/5, completeness 4/5, hallucination 2/5.
- An Oracle experiment manually supplied the decisive evidence about static edges while keeping the question and answer generation unchanged.
- With the decisive evidence, correctness, groundedness, and completeness reached 5/5; hallucination fell to 0/5.

**Finding:** clear evidence gap. This showed the model could solve the question when the evidence was available; it did not demonstrate that a new agent was necessary.

### Case 2 — Extensibility

The question asked how to design a LangGraph application that could accept new tools, agents, and workflows without modifying the existing graph structure.

Query expansion increased Hit Rate to 1.0, but the answer remained at 4/5 for correctness, groundedness, and completeness, with hallucination at 2/5.

**Finding:** better retrieval metrics did not improve the answer in this case. This was a useful control against assuming that broader retrieval always helps.

### Case 3 — Multi-tenant scalability

The question covered tenant isolation, `thread_id`, checkpointers, long-term memory stores, and concurrent users.

Query expansion retrieved the distinction between thread-scoped graph state and cross-thread long-term data. Correctness, groundedness, and completeness improved from 4/5 to 5/5; hallucination fell from 2/5 to 0/5.

| Metric | Original V6 | Query-expanded |
|---|---:|---:|
| Latency | 27.95 s | 41.66 s |
| Cost | $0.00218 | $0.00307 |
| Recall — RRF results | 0.333 | 0.500 |
| Precision — reranked selections | 0.000 | 0.167 |
| Hit Rate | 0.000 | 1.000 |
| MRR | 0.000 | 0.167 |

**Finding:** query expansion closed an evidence gap, but latency rose about 49.1% and cost about 41.0%.

### Case 4 — AI coding assistant architecture

The question combined planning, tool execution, human approval before code modification, checkpointing, and recovery from failed tool execution.

Query expansion improved correctness, groundedness, and completeness from 4/5 to 5/5 and reduced hallucination from 2/5 to 0/5.

| Metric | Original V6 | Query-expanded |
|---|---:|---:|
| Latency | 29.40 s | 49.75 s |
| Cost | $0.00270 | $0.00359 |
| Recall — RRF results | 0.333 | 0.714 |
| Precision — reranked selections | 0.400 | 0.500 |
| Hit Rate | 1.000 | 1.000 |
| MRR | 1.000 | 0.200 |

**Finding:** answer quality improved, but latency rose about 69.2%, cost about 33.0%, and MRR decreased. More evidence improved coverage while reducing early ranking quality and efficiency.

### Case 5 — Multi-agent research system

The question asked for a planning, retrieval, summarization, and review workflow.

Query expansion improved correctness, groundedness, and completeness from 4/5 to 5/5; hallucination fell from 2/5 to 0/5. Recall for RRF results increased from 0.333 to 0.600, while precision for reranked selections decreased from 0.333 to 0.167 and MRR from 1.0 to 0.5. Latency rose from 31.64 s to 39.92 s and cost from $0.00282 to $0.00320.

**Finding:** answer quality improved even though some retrieval metrics became worse.

### Case 6 — Graph construction and tool execution

The question asked for a graph review where tools should run only when the model requests a tool call.

Query expansion left correctness and completeness at 4/5, improved groundedness from 4/5 to 5/5, and reduced hallucination from 1/5 to 0/5. Recall for RRF results fell from 1.000 to 0.400, precision for reranked selections from 0.500 to 0.200. Latency rose from 26.42 s to 36.64 s and cost from $0.00186 to $0.00288.

**Finding:** neutral overall answer-quality result, with better grounding despite worse retrieval metrics.

### Case 7 — Architecture review / single large node

The question examined a design combining planning, tool execution, summarization, and final response generation in one node.

Query expansion improved correctness, groundedness, and completeness from 4/5 to 5/5; hallucination fell from 2/5 to 0/5.

| Metric | Original V5 | Query-expanded |
|---|---:|---:|
| Latency | 35.94 s | 45.62 s |
| Cost | $0.00303 | $0.00384 |
| Recall — RRF results | 0.400 | 0.800 |
| Precision — reranked selections | 0.000 | 0.500 |
| Hit Rate | 0.000 | 1.000 |
| MRR | 0.000 | 0.500 |

**Finding:** one of the strongest cases: retrieval coverage and answer quality improved, at higher latency and cost.

### V6 diagnostic-study takeaway

The results were mixed:

- Some difficult questions had decisive evidence missing from initial retrieval.
- Query expansion materially improved some answers.
- In other cases, retrieval metrics changed without a corresponding answer-quality improvement.
- Some query-expanded runs improved grounding even when recall, precision, or MRR declined.
- Broader retrieval often increased latency and cost.

These findings motivated more deliberate query strategies—including combined-query and multi-query approaches—and further investigation into whether iterative evidence gathering could address difficult multi-concept questions. They did not establish that every question needs more retrieval or that one strategy always wins.

---

## 3. Agentic RAG evaluation — 11 very difficult questions

Agentic RAG was evaluated on **11 very difficult questions** spanning code review, debugging, and design. The question set was created with AI assistance using questions and ideas gathered from around the internet. Relevant files were added to support evaluation, but those files are **not intended to be the only source of truth** for every answer: the questions can require synthesis, multi-step reasoning, and knowledge that is not fully stated in a single supplied file. For answer quality, the judge evaluation is therefore a central measure, considered alongside retrieval metrics and the evidence available to the system. This is a small, deliberately challenging set, not a statistically representative sample of all developer queries.

### Aggregate results

| Metric | Result |
|---|---:|
| Recall — RRF results | 0.7263 |
| Precision — reranked selections | 0.3771 |
| Hit Rate | 1.0000 |
| MRR | 0.5135 |
| Correctness | 4.9091 / 5 |
| Groundedness | 4.9091 / 5 |
| Completeness | 4.8182 / 5 |
| Hallucination | 0.0000 / 5 |
| Average latency | 91.25 s |
| Average cost per question | $0.01930 |
| Total cost for 11 questions | $0.21229 |

### Top-five per-question retrieval scores

**Precision scores** (reranked selections):
1. 0.5556
2. 0.5000
3. 0.5000
4. 0.4545
5. 0.4286

**Recall scores** (RRF results):
1. 0.8750
2. 0.8571
3. 0.8571
4. 0.8571
5. 0.8000

### Interpretation

On this hard 11-question set, Agentic RAG recorded Hit Rate 1.0, correctness and groundedness around 4.91/5, completeness around 4.82/5, and an average hallucination rating of 0/5. Average latency was about 91.25 seconds and average cost about $0.01930 per question.

The latency was measured **before the later concurrency and threading optimizations**. It is a pre-optimization measurement, not a claim about current latency.

### Comparison with the non-agentic pipeline

Before Agentic RAG, the non-agentic pipeline was tried on **some of the difficult questions**, rather than all 11. Those tests exposed substantial difficulty with multi-concept questions and questions requiring deeper reasoning. In those tried cases, judge ratings were approximately **2.5/5 for hallucination**, **3–4/5 for completeness**, and **3/5 for groundedness**. These are approximate observations from a subset of hard questions, not aggregate scores over the full 11-question set.

The non-agentic precision estimate of below 0.1 was also based on limited testing and is not presented here as a measured benchmark result. The key comparison is therefore qualitative: early tests showed that the non-agentic pipeline struggled on several hard, multi-concept questions, which helped motivate the Agentic RAG approach; the reported Agentic RAG aggregate below is measured on the 11-question challenge set.

---

## 4. What the experiments establish

### Supported conclusions

1. **Evidence coverage can be a bottleneck.** The command-routing Oracle experiment improved answer quality when decisive evidence was supplied.
2. **Query expansion can help difficult multi-concept questions.** Several cases improved in correctness, groundedness, completeness, and hallucination rating.
3. **Retrieval metrics do not substitute for answer evaluation.** Some answers improved while retrieval metrics declined; in another case, Hit Rate improved without improving answer quality.
4. **Iterative retrieval and revision are promising for hard questions.** The Agentic RAG run achieved high judged answer quality on the 11-question challenge set.
5. **There is a quality–efficiency trade-off.** Agentic RAG's recorded latency and cost were higher than those in the earlier per-question V6 examples, but the datasets and execution paths differ, so this is not an apples-to-apples comparison.
6. **Latency should be remeasured after optimization.** The 91.25-second average predates concurrency and threading changes.

### Limitations

- The V5 benchmark, V6 per-question diagnostics, and 11-question Agentic RAG evaluation have different scopes and must not be merged into one score table.
- Recall is measured on RRF results while precision is measured on reranked selections.
- Eleven difficult questions are useful for stress testing but do not establish general performance for all user queries.
- The non-agentic observations come from a subset of difficult questions, not a full evaluation across all 11; the stated ratings are approximate.
- The supplied supporting files are not an exhaustive source of truth for every question. Some questions require synthesis and reasoning beyond directly locating a fact in one file, so judge-based answer evaluation is important alongside retrieval metrics.
- Cost and latency depend on model settings, calls, runtime, and hardware.
- Judge scores should be read alongside per-question examples and evidence, not as proof of universal correctness.

---

## Summary

The project moved from a broad 82-question V5 baseline into targeted V6 failure analysis, which exposed evidence gaps and mixed query-expansion outcomes. These observations motivated Agentic RAG for difficult multi-concept questions. On a later set of 11 very difficult code-review, debugging, and design questions, Agentic RAG recorded high judged answer quality and a zero average hallucination rating, at a pre-concurrency-optimization latency of 91.25 seconds and cost of about $0.01930 per question.

Finally, the model was tested on the full benchmark_hard_questions.json, benchmark_debugging.json, and benchmark_code_review.json question sets, created with AI assistance using questions sourced from across the internet. Relevant supporting files were also included in the expected files for calculations, following the same evaluation approach used previously. However, the aggregate results across these complete benchmark sets have not yet been presented here. Instead, the final Agentic RAG evaluation reported above focuses on 11 separate, particularly difficult questions. These results indicate that the model can now handle multi-concept questions requiring deeper reasoning and evidence synthesis more effectively.

The engineering story is not that one pipeline is universally better. It is that the project used benchmark evidence to identify failure modes, tested retrieval interventions, measured trade-offs, and introduced iterative evidence gathering where difficult-question results justified the added complexity.
