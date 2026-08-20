# V7 — Retrieval / Agentic Capability Study

## Case 1 — Command Routing

### Question
`Command(goto=...)` + static `add_edge()` + checkpoint/resume behavior.

### Original V6

| Metric | Score |
|---|---:|
| Correctness | 4/5 |
| Groundedness | 4/5 |
| Completeness | 4/5 |
| Hallucination | 2/5 |

The answer correctly understood most `Command` and `resume` semantics, but could not establish the decisive interaction between `Command(goto=...)` and a statically defined edge.

### Oracle Experiment

The decisive evidence about static edges was manually supplied while keeping the question and answer generation unchanged.

| Metric | Score |
|---|---:|
| Correctness | **5/5** |
| Groundedness | **5/5** |
| Completeness | **5/5** |
| Hallucination | **0/5** |

### Conclusion

**Clear evidence gap.**

The LLM was capable of solving the problem. The original V6 retrieval simply failed to surface the decisive evidence.

**Classification:** Retrieval / evidence-selection failure.

**Agentic RAG required:** Not demonstrated.

---

# Case 2 — Extensibility

### Question

Design a LangGraph application where developers can add new tools, agents, and workflows without modifying the existing graph structure.

### Original V6

The original answer was already strong. It proposed:

- Stable core graph
- Separate tools
- Independent agents/workflows
- Stable dispatch layer
- Application-level registration
- Explicit caveats around undocumented registry/plugin behavior

It also made use of specific evidence around exported agents and configuration discovery.

### Query-Expanded Retrieval

The original question was supplemented with multiple focused retrieval queries and sent through the existing pipeline.

| Metric | Result |
|---|---:|
| Hit Rate | **1.0** |
| Recall@K | 0.60 |
| Precision@K | 0.167 |
| MRR | 0.167 |
| Correctness | 4/5 |
| Groundedness | 4/5 |
| Completeness | 4/5 |
| Hallucination | 2/5 |

The answer remained good, but query expansion did **not** improve the final answer. The evaluation continued to flag speculative claims around registration/registry mechanisms.

### Conclusion

**No demonstrated improvement.**

The retrieval hit rate increased to 1.0, but answer quality did not improve.

This demonstrates:

> **Better retrieval metrics do not necessarily produce a better final answer.**

**Classification:** No clear retrieval bottleneck demonstrated.

**Agentic RAG required:** No evidence.

---

# Comparison of First Two Cases

| | Case 1 — Command Routing | Case 2 — Extensibility |
|---|---|---|
| Original V6 | 4/5 | 4/5 |
| Query expansion | Not yet tested | No improvement |
| Oracle | **5/5** | Not tested |
| Main finding | **Missing decisive evidence** | No clear retrieval bottleneck |
| Agent required? | No evidence | No evidence |
| Important insight | Retrieval can be the bottleneck | Better retrieval ≠ better answer |

## Current Interpretation

After two cases:

### Case 1
Provides a **positive signal** that improving evidence retrieval may solve some hard questions.

### Case 2
Provides a **control case** where changing retrieval did not improve the answer.

This is useful because we are not assuming that query expansion works.

## Next Step

Continue with approximately **3–8 more genuinely difficult/weak V6 questions**.

For each:

1. Run the query-expanded version.
2. Compare against the existing V6 result.
3. Identify whether the improvement/failure comes from:
   - retrieval,
   - evidence integration,
   - reasoning,
   - or missing information.
4. Only after several cases decide whether query expansion should become part of V7.

### Current hypothesis

> **We should not build Agentic RAG yet. First determine whether the difficult tail of the benchmark can be solved by progressively stronger retrieval strategies.**


# Case 3 — Multi-Tenant Scalability

### Question

Design a multi-tenant LangGraph platform where each tenant's users have isolated workflow state and long-term memory. Explain how you would use checkpointers, `thread_id`, and stores to maintain isolation, and describe the considerations for serving many concurrent users.

### Original V6

| Metric | Score |
|---|---:|
| Correctness | 4/5 |
| Groundedness | 4/5 |
| Completeness | 4/5 |
| Hallucination | 2/5 |

The original answer correctly described:

- Checkpointers for thread-scoped workflow state
- `thread_id` for isolating workflow executions
- Persistent checkpointer usage
- Async invocation
- Production scaling considerations

However, the answer relied on a tenant-qualified `thread_id` convention such as:

`tenant_id:user_id:workflow_id`

The evaluator considered this an unsupported inference because the retrieved context did not explicitly document a tenant-isolation mechanism.

### Query-Expanded Retrieval

The original question was supplemented with multiple focused retrieval queries and sent through the existing pipeline.

| Metric | Score |
|---|---:|
| Correctness | **5/5** |
| Groundedness | **5/5** |
| Completeness | **5/5** |
| Hallucination | **0/5** |
| Missing concepts | **None** |
| Unsupported claims | **None** |

The query-expanded run retrieved the important persistence distinction:

- **Checkpointers** → thread-scoped graph state
- **Stores** → long-term, cross-thread application data

It also retrieved relevant deployment/scaling and checkpointing material. :contentReference[oaicite:0]{index=0} :contentReference[oaicite:1]{index=1}

The resulting answer was able to explain tenant/user isolation, the separation between workflow state and long-term memory, and concurrency considerations while explicitly distinguishing documented behavior from application-level architectural decisions. The evaluator found no unsupported claims. :contentReference[oaicite:2]{index=2}

### Improvement

| Metric | V6 → Query Expansion |
|---|---:|
| Correctness | **+1** |
| Groundedness | **+1** |
| Completeness | **+1** |
| Hallucination | **−2** |

| Metric | Original V6 | Query Expansion | Change |
|---|---:|---:|---:|
| Total latency | 27.95s | 41.66s | **+13.71s (+49.1%)** |
| Total cost | $0.00218 | $0.00307 | **+41.0%** |
| Recall@K | 0.333 | **0.500** | **+50%** |
| Precision@K | 0.000 | **0.167** | **Improved** |
| Hit rate | 0.000 | **1.000** | **Major improvement** |
| MRR | 0.000 | **0.167** | **Improved** |

### Conclusion

**Genuine positive result for query expansion.**

Unlike Case 2, where improved retrieval metrics did not improve the final answer, query expansion in this case produced a materially better answer.

The result suggests that the additional focused retrieval context helped the system distinguish between:

> **What LangGraph explicitly documents**

and

> **What must be implemented as application-level multi-tenant architecture.**

### Current Interpretation

This case provides evidence that **query expansion can close an evidence gap in difficult questions**.

However, with only three cases, it is still too early to conclude that query expansion should become V7.

---

# Current Three-Case Summary

| Case | Original V6 | Query Expansion / Oracle | Improvement? | Interpretation |
|---|---:|---:|---|---|
| **1. Command routing** | 4/5 | Oracle → **5/5** | ✅ Oracle improvement | Clear evidence gap |
| **2. Extensibility** | 4/5 | **4/5** | ❌ No | Better retrieval ≠ better answer |
| **3. Multi-tenant scalability** | 4/5 | **5/5** | ✅ Yes | Query expansion closed an evidence gap |

### Current takeaway

We now have **one clear query-expansion success, one neutral case, and one Oracle-confirmed retrieval failure**.

Continue testing the same approach on the remaining hard questions before deciding whether it deserves to become the next retrieval architecture.

# Case 4 — AI Coding Assistant Architecture

### Question

Design an AI coding assistant using LangGraph. Explain how the workflow should combine planning, tool execution, human approval before code modification, persistence/checkpointing, and recovery from failed tool executions. Distinguish documented LangGraph capabilities from application-level architectural decisions.

### Original V6

| Metric        | Score |
| ------------- | ----: |
| Correctness   |   4/5 |
| Groundedness  |   4/5 |
| Completeness  |   4/5 |
| Hallucination |   2/5 |

The original answer correctly described:

- Planning
- Tool execution with `ToolNode`
- Human approval before modifying code
- Persistence and checkpointing
- Recovery from failed tool executions
- A conceptual workflow combining these components

However, the evaluator identified unsupported claims around:

- Specific planning-step implementations
- The retry/recovery mechanism

The answer therefore received a hallucination score of **2/5**, despite covering the major concepts.

### Query-Expanded Retrieval

The question was supplemented with additional focused retrieval context and sent through the existing pipeline.

| Metric             |    Score |
| ------------------ | -------: |
| Correctness        |  **5/5** |
| Groundedness       |  **5/5** |
| Completeness       |  **5/5** |
| Hallucination      |  **0/5** |
| Missing concepts   | **None** |
| Unsupported claims | **None** |

The expanded retrieval provided relevant evidence covering:

- LangGraph's node/state workflow model
- Mixing deterministic and LLM-driven steps
- Human-in-the-loop interrupts
- Approval before critical actions
- Tool-call review and editing
- Persistence and resumption
- `ToolNode` capabilities
- Tool execution and validation errors
- Checkpoint durability modes

The resulting answer was also more careful about distinguishing documented capabilities from application-level inference. It explicitly stated when the retrieved context did not establish details such as the exact `ToolNode` API, state schema, retry mechanism, persistence constructor, or resume syntax. :contentReference[oaicite:0]{index=0}

The evaluator found **no unsupported claims and no missing concepts**. :contentReference[oaicite:1]{index=1}

### Improvement

| Metric        | V6 → Query Expansion |
| ------------- | -------------------: |
| Correctness   |               **+1** |
| Groundedness  |               **+1** |
| Completeness  |               **+1** |
| Hallucination |               **−2** |

| Metric        | Original V6 | Query Expansion |                Change |
| ------------- | ----------: | --------------: | --------------------: |
| Total latency |      29.40s |          49.75s | **+20.35s (+69.2%)** |
| Total cost    |    $0.00270 |        $0.00359 | **+33.0%** |
| Recall@K      |       0.333 |       **0.714** |          **+114.3%** |
| Precision@K   |       0.400 |       **0.500** |           **+25.0%** |
| Hit rate      |       1.000 |       **1.000** |                    — |
| MRR           |       1.000 |          0.200 |           **−80.0%** |

### Conclusion

**Genuine positive result for query expansion.**

The expanded retrieval substantially increased evidence coverage and improved the final answer from **4/5 → 5/5** across correctness, groundedness, and completeness.

The most important improvement was the reduction in unsupported claims:

> **Hallucination: 2/5 → 0/5**

The new answer was able to make a clearer distinction between:

> **What LangGraph explicitly documents**

and

> **What must be designed at the application level.**

However, this improvement came with a significant retrieval and execution cost: latency increased by approximately **69%**, cost increased by **33%**, and MRR dropped from **1.0 → 0.2**.

### Current Interpretation

Q46 provides **strong evidence that query expansion can improve difficult, multi-concept questions when the original retrieval does not provide enough supporting evidence**.

The interesting tradeoff is that broader evidence coverage came at the expense of early ranking quality and system efficiency.

This case therefore supports query expansion from an **answer-quality perspective**, but the latency, cost, and MRR regression need to be considered before adopting it as the next retrieval architecture.


# Case 5 — Multi-Agent Research System

### Question

Design a multi-agent research system where one agent plans, another retrieves information, a third summarizes findings, and a reviewer validates the final answer. Explain how the agents communicate and coordinate.

### Original V6

| Metric | Score |
|---|---:|
| Correctness | 4/5 |
| Groundedness | 4/5 |
| Completeness | 4/5 |
| Hallucination | 2/5 |

#### Retrieval Metrics

| Metric | Original V6 |
|---|---:|
| Total latency | 31.64s |
| Total cost | $0.00282 |
| Recall@K | 0.333 |
| Precision@K | 0.333 |
| Hit rate | 1.0 |
| MRR | 1.0 |

The original answer proposed a parent LangGraph workflow coordinating four specialized agents:

- Planner
- Retriever
- Summarizer
- Reviewer

It used shared state for communication, subgraphs for agent composition, conditional routing for reviewer decisions, and agentic RAG for retrieval.

The main weakness identified by the judge was that some specific implementation details and roles of the planner, summarizer, and reviewer were inferred rather than directly supported by the retrieved context.

### Query-Expanded Retrieval

The original question was supplemented with multiple focused retrieval queries.

#### Retrieval Metrics

| Metric | Original V6 | Query Expansion | Change |
|---|---:|---:|---:|
| Total latency | 31.64s | 39.92s | +26.2% |
| Total cost | $0.00282 | $0.00320 | +13.6% |
| Recall@K | 0.333 | **0.600** | **+80%** |
| Precision@K | 0.333 | 0.167 | -50% |
| Hit rate | 1.0 | 1.0 | No change |
| MRR | 1.0 | 0.500 | -50% |

#### Answer Evaluation

| Metric | Original V6 | Query Expansion | Change |
|---|---:|---:|---:|
| Correctness | 4/5 | **5/5** | **+1** |
| Groundedness | 4/5 | **5/5** | **+1** |
| Completeness | 4/5 | **5/5** | **+1** |
| Hallucination | 2/5 | **0/5** | **−2** |

The query-expanded answer was judged to accurately describe the multi-agent architecture using nodes, shared state, and edges. All significant claims were considered grounded, and there were no missing concepts or unsupported claims.

### Key Finding

This case is particularly interesting because **answer quality improved even though some retrieval metrics became worse**.

```text
Original V6
    ↓
Recall@K = 0.333
Precision@K = 0.333
MRR = 1.0
    ↓
Answer = 4/5

Query Expansion
    ↓
Recall@K = 0.600
Precision@K = 0.167
MRR = 0.5
    ↓
Answer = 5/5

# Case 6 — Graph Construction / Tool Execution

### Question

Review a LangGraph graph where the developer wants tool execution to occur only when the model requests a tool call.

### Original V6

| Metric | Score |
|---|---:|
| Correctness | 4/5 |
| Groundedness | 4/5 |
| Completeness | 4/5 |
| Hallucination | 1/5 |

#### Retrieval Metrics

| Metric | Original V6 |
|---|---:|
| Total latency | 26.42s |
| Total cost | $0.00186 |
| Recall@K | 1.000 |
| Precision@K | 0.500 |
| Hit rate | 1.0 |
| MRR | 1.0 |

The original answer correctly identified the unconditional `assistant → tools` edge as the problem and recommended conditional routing.

However, it could not provide the exact conditional-edge implementation because the retrieved context did not contain the required API details.

The judge also identified a minor unsupported claim concerning `ToolNode` behavior.

---

### Query-Expanded Retrieval

| Metric | Score |
|---|---:|
| Correctness | 4/5 |
| Groundedness | **5/5** |
| Completeness | 4/5 |
| Hallucination | **0/5** |

#### Retrieval Metrics

| Metric | Original V6 | Query Expansion | Change |
|---|---:|---:|---:|
| Total latency | 26.42s | 36.64s | +38.7% |
| Total cost | $0.00186 | $0.00288 | +54.7% |
| Recall@K | 1.000 | 0.400 | **−60%** |
| Precision@K | 0.500 | 0.200 | **−60%** |
| Hit rate | 1.0 | 1.0 | No change |
| MRR | 1.0 | 1.0 | No change |

### Answer Impact

Query expansion **did not improve overall answer quality**:

- Correctness: **4 → 4**
- Completeness: **4 → 4**

However, it improved:

- Groundedness: **4 → 5**
- Hallucination: **1 → 0**

The new answer provided a more concrete conditional-routing implementation using `tools_condition`, and the judge found no unsupported claims.

### Conclusion

**Neutral result for query expansion.**

Despite significantly worse retrieval metrics, the final answer maintained its correctness and completeness while becoming better grounded and free of the previously identified unsupported claim.

This reinforces an important observation from the experiment:

> **Retrieval metrics do not directly predict final answer quality.**

The query-expanded run retrieved fewer of the expected relevant chunks according to Recall@K and Precision@K, but the retrieved evidence was still sufficient to produce a well-grounded answer.

### Classification

**Answer quality:** ➖ No improvement  
**Groundedness:** ✅ Improved  
**Hallucination:** ✅ Improved  
**Retrieval metrics:** ❌ Worse  
**Overall:** **Neutral**

---

# Updated Case Summary

| Case | V6 | Query Expansion / Oracle | Result |
|---|---:|---:|---|
| **1. Command routing** | 4/5 | Oracle → **5/5** | ✅ Evidence gap confirmed |
| **2. Extensibility** | 4/5 | **4/5** | ➖ No improvement |
| **3. Multi-tenant scalability** | 4/5 | **5/5** | ✅ Improved |
| **5. Multi-agent research** | 4/5 | **5/5** | ✅ Improved |
| **6. Tool graph construction** | 4/5 | **4/5** | ➖ Neutral; grounding improved |

### Current Interpretation

So far, query expansion has produced **mixed results**:

- It clearly improved some difficult questions.
- It had no effect on others.
- In some cases, retrieval metrics worsened without hurting answer quality.
- The Oracle experiment confirms that at least one V6 failure was caused by missing evidence.

We should continue testing the remaining **4–6 selected hard/weak questions** before deciding whether query expansion should become the next retrieval architecture.


# Case 7 — Architecture Review / Single Large Node

### Question

Review a LangGraph architecture where planning, tool execution, summarization, and final response generation are combined into a single node. Evaluate the design from maintainability, extensibility, and debugging perspectives and recommend improvements.

### Original V6

| Metric | Score |
|---|---:|
| Correctness | 4/5 |
| Groundedness | 4/5 |
| Completeness | 4/5 |
| Hallucination | 2/5 |

#### Retrieval Metrics

| Metric | Original V6 |
|---|---:|
| Total latency | 35.94s |
| Total cost | $0.00303 |
| Recall@K | 0.400 |
| Precision@K | 0.000 |
| Hit rate | 0.0 |
| MRR | 0.0 |

The original answer correctly identified the architectural problems caused by combining four responsibilities into one node:

- Planning
- Tool execution
- Summarization
- Final response generation

It recommended decomposing these responsibilities into separate nodes with dedicated state fields.

However, retrieval was weak, with **0 hit rate, 0 precision, and 0 MRR**. The judge also identified unsupported claims around debugging/observability.

---

### Query-Expanded Retrieval

| Metric | Score |
|---|---:|
| Correctness | **5/5** |
| Groundedness | **5/5** |
| Completeness | **5/5** |
| Hallucination | **0/5** |

#### Retrieval Metrics

| Metric | Original V6 | Query Expansion | Change |
|---|---:|---:|---:|
| Total latency | 35.94s | 45.62s | +26.9% |
| Total cost | $0.00303 | $0.00384 | +26.6% |
| Recall@K | 0.400 | **0.800** | **+100%** |
| Precision@K | 0.000 | **0.500** | **Improved** |
| Hit rate | 0.0 | **1.0** | **Major improvement** |
| MRR | 0.0 | **0.500** | **Improved** |

### Answer Impact

| Metric | V6 | Query Expansion | Change |
|---|---:|---:|---:|
| Correctness | 4/5 | **5/5** | **+1** |
| Groundedness | 4/5 | **5/5** | **+1** |
| Completeness | 4/5 | **5/5** | **+1** |
| Hallucination | 2/5 | **0/5** | **−2** |

The query-expanded answer provided a more comprehensive and grounded architectural analysis, covering:

- Explicit node boundaries
- Shared state between stages
- Workflow vs. agent design
- `ToolNode` for dedicated tool execution
- Conditional routing and loops
- Maintainability and extensibility
- Debugging implications

The judge found **no missing concepts and no unsupported claims**.

### Key Finding

This is one of the strongest query-expansion results so far because **retrieval quality and answer quality improved together**.

```text
Original V6
    ↓
Recall@K = 0.40
Precision@K = 0
Hit rate = 0
MRR = 0
    ↓
Answer = 4/5

Query Expansion
    ↓
Recall@K = 0.80
Precision@K = 0.50
Hit rate = 1.0
MRR = 0.50
    ↓
Answer = 5/5



