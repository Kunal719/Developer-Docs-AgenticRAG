from pathlib import Path

PROJECT_VERSION = "v5"

GITHUB_DOC_REPO_URL = "https://github.com/langchain-ai/docs.git"
LOCAL_DOC_REPO_PATH =  Path("data/raw/docs")
DOCUMENTATION_PATH = Path("data/raw/docs/src/oss/langgraph")

GITHUB_CODE_REPO_URL = "https://github.com/langchain-ai/langgraph.git"
LOCAL_CODE_REPO_PATH = Path("data/raw/langgraph")
IMPLEMENTATION_PATH = Path("data/raw/langgraph/libs")

DOC_PERSISTENT_DIRECTORY = Path("data/chromadb/langgraph_docs")
CODE_PERSISTENT_DIRECTORY = Path("data/chromadb/langgraph_code")
DOC_COLLECTION_NAME = "langgraph_docs"
CODE_COLLECTION_NAME = "langgraph_code"

EMBEDDING_MODEL = "text-embedding-3-small"

DOC_INDEX_PATH = Path("data/doc_index_metadata.json")
CODE_INDEX_PATH = Path("data/code_index_metadata.json")

DOC_BM25_PATH = Path("data/bm25/bm25_docs_index.pkl")
CODE_BM25_PATH = Path("data/bm25/bm25_code_index.pkl")

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

DENSE_TOP_K = 10
BM25_TOP_K = 10
DOC_HYBRID_TOP_K = 12
CODE_HYBRID_TOP_K = 16
RRF_K = 60

MULTI_QUERY_TOP_K = 40

RERANKER_MODEL = "BAAI/bge-reranker-base"
RERANK_CANDIDATES = 15
RERANK_TOP_K = 13

MODEL_NAME = "gpt-5.6-luna"
MODEL_PROVIDER = "openai"
MODEL_TEMP = 0.1

JUDGE_MODEL_NAME = "gpt-4o-mini"
JUDGE_MODEL_PROVIDER = "openai"
JUDGE_TEMP = 0.0

REBUILD_INDEX = False

GENERATION_INPUT_PRICE = 0.2
GENERATION_OUTPUT_PRICE = 1.2

JUDGE_INPUT_PRICE = 0.3
JUDGE_OUTPUT_PRICE = 2.5

DOCUMENTATION_SPARSE_PATHS = [
    "src/oss/langgraph",
]

IMPLEMENTATION_SPARSE_PATHS = [
    # Graphs
    "libs/langgraph/langgraph/graph/state.py",
    "libs/langgraph/langgraph/graph/message.py",
    "libs/langgraph/langgraph/graph/_node.py",
    "libs/langgraph/langgraph/graph/_branch.py",

    # Functional API
    "libs/langgraph/langgraph/func/__init__.py",

    # Pregel
    "libs/langgraph/langgraph/pregel/main.py",
    "libs/langgraph/langgraph/pregel/_checkpoint.py",
    "libs/langgraph/langgraph/pregel/_loop.py",
    "libs/langgraph/langgraph/pregel/_algo.py",
    "libs/langgraph/langgraph/pregel/_read.py",
    "libs/langgraph/langgraph/pregel/_write.py",
    "libs/langgraph/langgraph/pregel/_runner.py",
    "libs/langgraph/langgraph/pregel/_call.py",
    "libs/langgraph/langgraph/pregel/_executor.py",
    "libs/langgraph/langgraph/pregel/_retry.py",

    # Checkpointing
    "libs/checkpoint/langgraph/checkpoint/base/__init__.py",
    "libs/checkpoint/langgraph/checkpoint/base/id.py",
    "libs/checkpoint/langgraph/checkpoint/serde/base.py",
    "libs/checkpoint/langgraph/checkpoint/serde/jsonplus.py",
    "libs/checkpoint/langgraph/checkpoint/serde/encrypted.py",
    "libs/checkpoint/langgraph/checkpoint/memory/__init__.py",
    "libs/checkpoint-sqlite/langgraph/checkpoint/sqlite/__init__.py",
    "libs/checkpoint-sqlite/langgraph/checkpoint/sqlite/aio.py",
    "libs/checkpoint-postgres/langgraph/checkpoint/postgres/__init__.py",
    "libs/checkpoint-postgres/langgraph/checkpoint/postgres/aio.py",

    # Storage
    "libs/langgraph/langgraph/managed/base.py",
    "libs/checkpoint/langgraph/store/base/__init__.py",
    "libs/checkpoint/langgraph/store/base/batch.py",
    "libs/checkpoint/langgraph/store/base/embed.py",
    "libs/checkpoint/langgraph/store/memory/__init__.py",
    "libs/checkpoint-postgres/langgraph/store/postgres/base.py",
    "libs/checkpoint-postgres/langgraph/store/postgres/aio.py",

    # Types
    "libs/langgraph/langgraph/types.py",

    # Runtime
    "libs/langgraph/langgraph/runtime.py",

    # Config
    "libs/langgraph/langgraph/config.py",

    # Errors
    "libs/langgraph/langgraph/errors.py",

    # Channels
    "libs/langgraph/langgraph/channels/base.py",
    "libs/langgraph/langgraph/channels/topic.py",
    "libs/langgraph/langgraph/channels/last_value.py",
    "libs/langgraph/langgraph/channels/ephemeral_value.py",
    "libs/langgraph/langgraph/channels/any_value.py",
    "libs/langgraph/langgraph/channels/binop.py",
    "libs/langgraph/langgraph/channels/delta.py",

    # Prebuilt / Agents
    "libs/prebuilt/langgraph/prebuilt/tool_node.py",
    "libs/prebuilt/langgraph/prebuilt/_tool_call_transformer.py",
    "libs/prebuilt/langgraph/prebuilt/_tool_call_stream.py",
    "libs/prebuilt/langgraph/prebuilt/interrupt.py",

    # Internal
    "libs/langgraph/langgraph/_internal/_config.py",
    "libs/langgraph/langgraph/_internal/_replay.py"

]