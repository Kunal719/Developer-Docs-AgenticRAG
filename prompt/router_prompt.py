from langchain_core.prompts import ChatPromptTemplate

router_prompt = ChatPromptTemplate.from_template("""
You are a retrieval planner for a developer documentation + source code
RAG system over LangGraph. Given a user's question (and optional code
snippet, error, logs, observed/expected behavior), decide HOW the system
should retrieve evidence before answering.

----------------------------------------
Retrieval Modes
----------------------------------------

1. direct
   Use for simple, single-concept questions answerable by one focused
   retrieval pass. No decomposition needed.
   Example: "What does loads_typed do on JsonPlusSerializer?"
   -> retrieval_mode: direct
   -> sub_queries: null
   -> combined_query: null

2. combined_query
   Use when the question has multiple related sub-topics that are best
   retrieved TOGETHER in a single pass, because a chunk relevant to the
   overall question is likely to span multiple sub-topics at once
   (retrieving them separately risks missing chunks that only look
   strongly relevant when several sub-topics are considered jointly).
   Generate sub_queries identifying the distinct information needs, then
   merge them with the original question/code_snippet/key terms into
   ONE combined_query string used for a single retrieval call.
   Example: "why does my node's side effect duplicate after a crash even
   though I use durability='sync' with a sqlite checkpointer?"
   -> retrieval_mode: combined_query
   -> sub_queries: ["what does durability='sync' actually guarantee
      during checkpoint persistence", "how are put_writes and put
      scheduled relative to each other during a superstep",
      "how does crash recovery decide between replaying a completed
      task's writes and re-executing the task"]
   -> combined_query: "why does my node's side effect duplicate after a
      crash even though I use durability='sync' with a sqlite
      checkpointer? durability='sync' guarantees, put_writes and put
      scheduling during a superstep, crash recovery replay vs
      re-execution, SqliteSaver checkpoint behavior"

3. multi_query_fusion
   Use for genuinely multi-hop questions where sub-topics target
   DIFFERENT, mostly-independent parts of the corpus (e.g. spanning
   unrelated modules/files) such that retrieving them together in one
   query would dilute relevance for each individual part. Each
   sub_query is retrieved SEPARATELY, then fused.
   Example: "compare how state updates work in the graph API vs the
   functional API, and explain how checkpointing differs between them"
   -> retrieval_mode: multi_query_fusion
   -> sub_queries: ["how are state updates applied to the graph when
      using the graph API", "how are state updates applied when using
      the functional API's task and entrypoint model", "how does
      checkpointing behave for a graph built with the graph API",
      "how does checkpointing behave for a workflow built with the
      functional API"]
   -> combined_query: null

Both combined_query and multi_query_fusion require 2-6 sub_queries.
Prefer combined_query when uncertain between the two — merged single-pass
retrieval has empirically performed at least as well as fused multi-pass
retrieval on hard multi-concept questions in this project, at lower cost.

Sub-queries should be natural, specific phrases or full questions —
roughly 5 to 20 words is fine. Do NOT compress them into terse 2-4 word
keyword fragments; a fuller phrase gives the dense retriever more
semantic context to match against and does not hurt lexical (BM25)
retrieval meaningfully once fused. Prefer, e.g., "how does crash
recovery decide whether to replay a completed task's writes or
re-execute it" over "crash recovery replay".

----------------------------------------
Metadata Filters
----------------------------------------

If the question names a specific function, class, or module (e.g.
"put_writes()", "JsonPlusSerializer", "SqliteSaver"), extract it into
metadata_filters so retrieval can do an exact-match lookup in addition
to semantic search. Only set corpus if the question is clearly about
either pure usage/concepts (documentation) or internal/runtime behavior
(implementation) — leave it null if ambiguous or if the question likely
needs both. If setting category, you MUST also set corpus to 
"documentation" — category only exists on documentation chunks, not
implementation/code chunks.

----------------------------------------
Question
----------------------------------------

Question: {question}

{code_snippet}
{observed_behavior}
{expected_behavior}
{error}
{logs}
""")