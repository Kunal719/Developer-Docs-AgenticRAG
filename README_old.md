# Developer Docs RAG

A production-inspired Retrieval-Augmented Generation (RAG) project that answers developer questions using **official documentation** instead of relying solely on an LLM's internal knowledge.

Unlike many RAG tutorials that stop after indexing a set of documents into a vector database, this repository is intentionally built as an **iterative engineering project**. Each version introduces one architectural improvement while preserving and extending the previous implementation, allowing the system to evolve from a simple Dense RAG pipeline into a production-oriented developer documentation assistant.

---

# Why This Project?

Most RAG examples demonstrate the retrieval pipeline but assume the knowledge base is static.

Real-world systems are different.

Documentation changes continuously, repositories evolve, retrieval quality becomes increasingly important, and systems must be evaluated before introducing more advanced retrieval strategies.

The goal of this repository is to explore that evolution step by step by implementing production-inspired improvements rather than building multiple disconnected RAG projects.

Each version focuses on a single architectural enhancement while maintaining a clean, modular codebase.

This makes it possible to evaluate the impact of every change independently rather than combining multiple techniques into a single iteration. The project evolves through measured engineering decisions supported by benchmarking rather than feature accumulation.

---

# Evaluation Philosophy

Each architectural improvement in this project is evaluated before moving to the next iteration.

Rather than assuming a new retrieval strategy improves the system, every version is benchmarked independently using the project's evaluation framework. This approach makes it possible to understand not only whether a change helps, but also where it helps and where it introduces new trade-offs.

Detailed benchmark reports and experiment outputs are available in the `evaluation/` directory.

---

# Current Implementation

The project currently consists of six completed milestones.

## ✅ Version 1 — Dense RAG

Version 1 established the complete Dense RAG pipeline from scratch.

Features include:

* GitHub repository ingestion
* Markdown / MDX parsing
* Metadata extraction
* Recursive document chunking
* OpenAI embeddings (`text-embedding-3-small`)
* Persistent Chroma vector database
* Dense semantic retrieval
* LCEL-based RAG pipeline
* Interactive CLI

---

## ✅ Version 2 — Incremental Indexing

Version 2 focuses on making the ingestion pipeline significantly more efficient.

Instead of rebuilding the vector database whenever documentation changes, the application updates only the affected documents.

Version 2 introduces:

* Sparse checkout repository synchronization
* SHA-256 document change detection
* Metadata index for tracked documents
* Detection of:

  * Newly added documents
  * Modified documents
  * Deleted documents
* Selective document loading
* Selective vector deletion
* Incremental embedding generation
* Persistent Chroma updates
* Automatic indexing skip when documentation has not changed

The retrieval pipeline remains unchanged while the indexing pipeline becomes significantly more efficient.

---

## ✅ Version 3 — Hybrid Retrieval

Version 3 focuses on improving retrieval quality by combining dense semantic search with sparse lexical search.

Instead of relying solely on vector similarity, the retrieval pipeline now fuses results from both retrieval strategies using Reciprocal Rank Fusion (RRF), improving recall for API names, function names, and exact developer terminology while preserving semantic search capabilities.

Version 3 introduces:

* Dense Retriever abstraction
* BM25 lexical retrieval
* Hybrid Retriever
* Reciprocal Rank Fusion (RRF)
* Unified retrieval interface
* Improved retrieval for API-centric queries
* Prompt refinement to encourage evidence-based synthesis across retrieved documentation

---

## ✅ Version 4 — Cross-Encoder Reranking & Evaluation

Version 4 introduces a Cross-Encoder reranking stage to improve retrieval quality after Hybrid Retrieval.

Rather than assuming this architectural change improves the system, the project now includes a dedicated evaluation framework for benchmarking retrieval strategies across different versions.

Version 4 introduces:

* Cross-Encoder document reranking
* Modular evaluation framework
* Experiment metadata tracking
* Pipeline stage logging
* Structured JSON experiment outputs
* Benchmark question framework
* Manual evaluation workflow
* Benchmark reports and observations

---

## ✅ Version 5 — Multi-Corpus Retrieval

Version 5 expands the knowledge base beyond the official documentation by introducing multi-corpus retrieval.

Instead of searching only the LangGraph documentation, the retrieval pipeline now combines evidence from multiple sources before reranking and generation.

Version 5 introduces:

* Official LangGraph documentation corpus
* LangGraph API reference corpus
* LangGraph implementation source code corpus
* Unified multi-corpus retrieval
* Cross-corpus reranking
* Implementation-aware question answering
* Support for deep runtime and source-level reasoning

This enables the system to answer implementation-focused developer questions that cannot be answered reliably using documentation alone.

---

## ✅ Version 6 — Evaluation & Observability

Version 6 focuses on measuring the system rather than simply adding another retrieval technique.

The system is evaluated using **82 benchmark questions** across:

* General developer questions
* Code review
* Debugging

Version 6 introduces:

* Recall@K
* Precision@K
* Hit Rate@K
* Mean Reciprocal Rank (MRR)
* LLM-as-Judge evaluation
* Correctness, groundedness, completeness and hallucination scoring
* Pipeline stage observability
* Generation token tracking
* API cost tracking
* End-to-end latency measurement
* Structured benchmark result JSON
* FastAPI backend
* Streamlit interface

The evaluation framework is used to make architectural decisions based on measured failure cases rather than adding complexity automatically.

---

# Project Overview

The knowledge base for this project consists of two corpuses:

* Official LangGraph documentation
* Carefully selected LangGraph API Reference/implementation source code

The retrieval pipeline searches across all corpora, combines the retrieved candidates, reranks them using a Cross-Encoder, and generates answers grounded in both documentation and implementation details.

The application performs two independent workflows.

## Indexing Pipeline

The indexing pipeline keeps the local vector database synchronized with the latest documentation.

```text
GitHub Repository
        │
        ▼
Sparse Checkout Synchronization
        │
        ▼
Repository Status
        │
 ┌──────┴─────────────┐
 │                    │
 │              Repository Updated
 │                    │
 │                    ▼
 │           SHA-256 Change Detection
 │                    │
 │                    ▼
 │        Delete Updated / Removed Vectors
 │                    │
 │                    ▼
 │          Load Changed Documents
 │                    │
 │                    ▼
 │          Metadata Extraction
 │                    │
 │                    ▼
 │               Document Chunking
 │                    │
 │                    ▼
 │          OpenAI Embedding Model
 │                    │
 │                    ▼
 │      Persistent Chroma Vector Store
 │
 ▼
Repository Unchanged
        │
        ▼
Skip Indexing
```

---

## Retrieval Pipeline

```text
User Question
      │
      ▼
Multi-Corpus Retrieval
      │
 ┌────┼─────────────┐
 │    │             │
 ▼    ▼             ▼
Documentation   API Reference   Implementation
 │    │             │
 └────┴─────────────┘
        │
        ▼
Cross-Encoder Reranker
        │
        ▼
Top Retrieved Context
        │
        ▼
Prompt Template
        │
        ▼
OpenAI Chat Model
        │
        ▼
Generated Answer
```

---

## Evaluation Pipeline

```text
Benchmark Question
        │
        ▼
Retrieval Pipeline
        │
        ▼
Generation
        │
        ▼
Capture Pipeline Stages
        │
        ▼
Capture Generation Metrics
        │
        ▼
Capture Overall Metrics
        │
        ▼
Experiment JSON
        │
        ▼
LLM-as-Judge Evaluation
```

# Project Structure

```text
developer-docs-rag/
│
├── app/
├── ingestion/
├── retrieval/
├── reranking/
├── pipeline/
├── prompts/
├── evaluation/
│   └── results/
├── data/
│   ├── chromadb/
│   ├── bm25/
│   └── raw/
├── tests/
├── utils/
│
├── main.py
├── requirements.txt
├── README.md
└── .env
```

---

# Design Principles

The project follows several design principles that make future versions easier to build.

* Modular ingestion pipeline
* Separation of indexing and retrieval
* Persistent vector database
* Incremental document synchronization
* Version-by-version evolution
* Production-inspired architecture over notebook prototypes

---

# Tech Stack

* Python 3.12
* LangChain
* ChromaDB
* OpenAI
* GitPython
* python-frontmatter
* rank-bm25
* sentence-transformers
* FlagEmbedding / BAAI BGE Reranker
* FastAPI
* Uvicorn
* Streamlit

---

# Example

<img width="1917" height="973" alt="image" src="https://github.com/user-attachments/assets/55c7ad9f-87c8-460b-9466-a9579d487ad3" />

<img width="1450" height="837" alt="image" src="https://github.com/user-attachments/assets/76d44a3a-762c-4719-aee4-6ab4b403b320" />

---

# Improvements Introduced in Version 2

Compared to Version 1, the indexing pipeline now:

* Avoids reparsing unchanged documentation
* Avoids re-embedding unchanged documents
* Detects added, modified and deleted documentation
* Removes obsolete vectors before re-indexing
* Updates only affected documents
* Skips indexing completely when the documentation repository has not changed

These improvements significantly reduce startup work while preserving the same retrieval behaviour.

---

# Improvements Introduced in Version 3

Compared to Version 2, the retrieval pipeline now:

* Combines dense semantic retrieval with BM25 lexical retrieval
* Uses Reciprocal Rank Fusion (RRF) to merge retrieval results
* Improves retrieval of exact API names and developer terminology
* Introduces a modular retrieval architecture for future retrieval strategies
* Encourages synthesis across multiple retrieved documentation chunks while remaining grounded in the retrieved context

These improvements increase retrieval robustness while maintaining the existing indexing pipeline introduced in Version 2.

---

# Improvements Introduced in Version 4

Compared to Version 3, the retrieval pipeline now:

* Adds Cross-Encoder reranking after Hybrid Retrieval.
* Improves ranking of semantically relevant documentation chunks.
* Introduces a modular evaluation framework for benchmarking retrieval strategies.
* Captures experiment metadata and pipeline stage metrics.
* Produces structured experiment JSON for reproducible analysis.
* Enables systematic comparison between retrieval pipeline versions.

These improvements make retrieval evaluation a first-class component of the project rather than relying solely on qualitative inspection.

---

# Improvements Introduced in Version 5

Compared to Version 4, the retrieval pipeline now:

* Introduces multi-corpus retrieval.
* Indexes the LangGraph API reference.
* Indexes the LangGraph implementation source code.
* Retrieves evidence across documentation, API reference, and implementation.
* Improves implementation-level question answering.
* Enables source-code grounded explanations for internal runtime behaviour.

These improvements significantly expand the knowledge available to the retrieval pipeline while preserving the retrieval architecture introduced in Version 4.

---

# Project Roadmap

## ✅ Version 1 — Dense RAG

* GitHub ingestion
* Markdown parsing
* Metadata extraction
* Recursive chunking
* Dense retrieval
* Persistent Chroma vector database
* LCEL RAG pipeline
* Interactive CLI

---

## ✅ Version 2 — Incremental Indexing

* Sparse repository synchronization
* SHA-256 document change detection
* Metadata index
* Selective document loading
* Selective vector deletion
* Incremental embedding updates
* Skip indexing when documentation is unchanged

---

## ✅ Version 3 — Hybrid Retrieval

* BM25
* Dense retrieval
* Reciprocal Rank Fusion (RRF)

---

## ✅ V4 Cross Encoder + Evaluation

Improve retrieval quality by reranking retrieved documents before generation.

---

## ✅ Version 5 — Multi-Corpus Retrieval

Expand the knowledge base by combining official documentation with the LangGraph API source code.

* Official documentation
* API source code
* Rich metadata
* Multi-corpus retrieval

---

## ✅ Version 6 — Evaluation & Observability

* 82-question benchmark
* Recall@8
* Precision@8
* Hit Rate@8
* MRR
* LLM-as-Judge
* Correctness / groundedness / completeness / hallucination
* Token usage
* Cost tracking
* End-to-end latency
* Structured benchmark results
* FastAPI backend
* Streamlit interface

---

## ⬜ Next — Evidence-Driven Retrieval Improvements

The next architectural direction is intentionally not fixed yet.

The V6 benchmark will be used to determine whether the system actually benefits from:

* Query analysis
* Adaptive retrieval
* Multi-step retrieval
* Evidence verification
* Query rewriting
* Agentic retrieval

Agentic RAG will only be introduced if the measured failure cases justify the additional complexity.

---

# Installation

```bash
git clone <repository-url>
cd developer-docs-rag
```

Create a virtual environment.

```bash
python -m venv .venv
```

Activate it.

**Windows**

```bash
.venv\Scripts\activate
```

Install dependencies.

```bash
pip install -r requirements.txt
```

Create a `.env` file.

```env
OPENAI_API_KEY=your_api_key
```

Run the application on CLI.

```bash
python main.py
```
### Run FastAPI

```bash
uvicorn app.api.main:app --reload
```

### Run Streamlit

```bash
streamlit run ui/streamlit_app.py
```

---

# Version 5 Evaluation

Version 5 was evaluated using **82 benchmark questions** covering:

* General developer questions — 52
* Code review — 20
* Debugging — 10

### Overall Results

| Metric | Result |
|---|---:|
| Recall@8 | **0.624** |
| Precision@8 | **0.396** |
| Hit Rate@8 | **0.902** |
| MRR | **0.743** |
| Correctness | **4.756 / 5** |
| Groundedness | **4.793 / 5** |
| Completeness | **4.756 / 5** |
| Hallucination | **0.305 / 5** |
| Average end-to-end latency | **28.97s** |
| Average token usage | **3,472 tokens/question** |
| Average cost | **~$0.00214/question** |

Latency was measured end-to-end on a CPU-only development machine.

The results show a useful distinction between retrieval and answer quality: retrieval is not perfect, particularly for broad code-review and architectural questions, while generated answers are generally strong and well-grounded.

The benchmark is therefore treated as a baseline for future architectural decisions rather than as a target to optimize blindly. Before introducing additional complexity such as Agentic RAG, the failure cases should be investigated to determine whether the underlying issue is retrieval coverage, ranking, benchmark expectations, or the need for iterative evidence gathering.

Detailed experiment outputs are available in the `evaluation/results/` directory.

# Future Improvements

Future work will be driven by the V6 benchmark rather than by adding components automatically.

Possible extensions include:

* Evidence-driven adaptive retrieval
* Query analysis and rewriting
* Iterative / multi-step retrieval
* Evidence verification
* Retrieval and reranking latency optimization
* Background indexing
* Automatic documentation versioning
* Docker deployment
* CI/CD pipeline
* Streaming responses
* Conversation memory
* Expansion to additional LangChain documentation and implementation sources

The key design principle is to add complexity only when evaluation shows that the current architecture needs it.
