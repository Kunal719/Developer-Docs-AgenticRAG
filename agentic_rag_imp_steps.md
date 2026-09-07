# RAG Architecture Refactoring — Steps 1–4

## Step 1 — Introduce Explicit `RAGState`

### What we did

Created a dedicated `RAGState` to hold the mutable state of a RAG execution.

It contains:

* `query_input`
* `retrieval_plan`
* `reranker_query`
* `initial_evidence`
* `evidence_pool`
* `final_evidence`
* `generated_answer`
* `gaps`
* `generation_round`
* `stage`

We also introduced `RAGStage` to explicitly represent pipeline progress:

```text
INITIALIZED
    ↓
ROUTED
    ↓
INITIAL_EVIDENCE_READY
    ↓
DRAFT_GENERATED
    ↓
GAPS_PROCESSED
    ↓
FINAL_EVIDENCE_READY
    ↓
COMPLETED
```

### Why

Previously, execution state was distributed across local variables inside `run()`.

This made the pipeline harder to reason about and would make checkpointing difficult later.

With `RAGState`, the pipeline has an explicit representation of its current state and progress.

It also prepares the system for future checkpointing without requiring LangGraph.

---

# Step 2 — Introduce `EvidenceItem` and Provenance

### What we did

Previously, evidence was represented only as:

```python
list[Document]
```

We introduced an `EvidenceItem`:

```python
@dataclass
class EvidenceItem:
    document: Document
    source: str
    query: str
    gap: RetrievalGap | None = None
```

Documents are wrapped **after reranking**.

For initial retrieval:

```text
Retrieve Documents
      ↓
Rerank
      ↓
EvidenceItem(
    source="initial"
)
```

For gap retrieval:

```text
Retrieve Documents
      ↓
Rerank
      ↓
EvidenceItem(
    source="gap",
    gap=<specific gap>
)
```

### Why

Once evidence from multiple retrieval paths enters the same evidence pool, we need to know:

* where the evidence came from
* which query produced it
* whether it was retrieved for a specific gap

This provenance becomes important for later gap-aware ranking and coverage-preserving selection.

Importantly, retrieval and reranking remain unaware of provenance.

The orchestration layer attaches provenance because it is the layer that knows the semantic origin of the evidence.

---

# Step 3 — Introduce `EvidenceSelector`

### What we did

Moved evidence-selection responsibility behind an abstraction:

```text
EvidenceSelector
       ↓
RerankEvidenceSelector
```

`AgenticRAG` receives an `EvidenceSelector` through dependency injection instead of implementing the selection logic itself.

The current selector performs:

```text
EvidenceItems
      ↓
Deduplication
      ↓
Reranking
      ↓
Top-K Selection
```

The selector also accepts `PipelineObserver` and passes it to the reranker.

### Why

The orchestrator should coordinate the pipeline rather than contain the details of evidence-selection policies.

The abstraction gives us a clean extension point for future selection strategies, particularly:

* gap-aware reranking
* gap-coverage preservation
* other evidence-selection policies

Eventually, `AgenticRAG` should only need to express:

> Select the best evidence from this evidence pool.

without knowing the internal selection policy.

---

# Step 4 — Evidence Deduplication

### What we did

Kept the existing document-level `_dedupe()` for raw retrieval candidates.

Added a separate evidence-level deduplication function:

```python
dedupe_evidence(evidence: list[EvidenceItem])
```

The evidence-level deduplication currently considers:

1. Exact `chunk_id`
2. Normalized content

The architecture is:

```text
Raw Retrieval
    ↓
Document-level dedupe
    ↓
Reranking
    ↓
EvidenceItem + provenance
    ↓
Evidence-level dedupe
```

### Why two deduplication stages?

The raw retrieval layer still works with:

```python
list[Document]
```

while the evidence-selection layer works with:

```python
list[EvidenceItem]
```

They therefore have different responsibilities.

The early document-level dedupe prevents obvious duplicate retrieval candidates.

The later evidence-level dedupe handles duplicates after provenance has been attached and before final evidence selection.

### Chunking Considerations

Documentation chunks are created using:

```python
chunk_size = 1200
chunk_overlap = 200
```

Therefore, adjacent documentation chunks can legitimately share text.

Code chunks are different: they are generated at the AST level, with one function/method generally representing one chunk and no artificial chunk overlap.

Because of this, we did **not** introduce an arbitrary fuzzy/embedding similarity threshold for deduplication.

Potential overlap-based deduplication can be added later if benchmark results demonstrate that it improves evidence quality without removing useful neighboring documentation chunks.

---

# Current Architecture

```text
Query
  ↓
RAGState
  ↓
Query Router
  ↓
Retrieval
  ↓
Document-level Dedupe
  ↓
Reranking
  ↓
EvidenceItem + Provenance
  ↓
EvidenceSelector
  ├── Evidence-level Dedupe
  ├── Reranking
  └── Selection
  ↓
Final Evidence
  ↓
Generation
```

# Next Step

## Step 5 — Gap-Aware Global Reranking

The next change is to make final evidence selection consider:

1. The original user question
2. The unresolved retrieval gaps

This prevents evidence that is highly relevant to a specific gap from being pushed down simply because it is less relevant to the original question alone.

Step 5 will build on the `EvidenceSelector` abstraction introduced in Step 3.