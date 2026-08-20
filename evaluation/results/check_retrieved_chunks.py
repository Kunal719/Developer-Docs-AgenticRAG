from pathlib import Path
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

doc_vector_store = Chroma(
    collection_name="langgraph_docs",
    persist_directory=str(Path("data/chromadb/langgraph_docs")),
    embedding_function=embedding_model,
)

code_vector_store = Chroma(
    collection_name="langgraph_code",
    persist_directory=str(Path("data/chromadb/langgraph_code")),
    embedding_function=embedding_model,
)

chunk_ids = [
    "ac1c2fc3bf7cfd682e154b0d783058fd3c75b865132aa4786c1b5ea9dd84",
    "11eb3972379a12b79c00359d25a8a244b95260ff36e7b17c41140342c1d17ea5",
    "12177d28d30df3df291dae98eafc1b573db6e5417f410ca53af56b5bf7e8f59b",
    "c0287e5e8975e762f465de5a9ccef1879006c8a0368ca7ffc38fb514ecbc6b72",
    "89151b67b607b8611de20511df4e4af0497df484fe6d01f6d65107c3852a64f3",
    "49cf0069b903cb1ce54dfb6ab3ec2a4528b78e2435c49a1aa5204e1d05997453",
    "ef26bca64344dcc1f5af7d1637886e7954c42c9b414f5a82ae50e41c4b76ee8a",
    "8888f39386ffc838b3d131f28c489e5e8d3720ff94b90cd58a321b11aa2ae552",
]

stores = [
    ("DOCUMENTATION", doc_vector_store),
    ("IMPLEMENTATION", code_vector_store),
]

for store_name, store in stores:
    results = store.get(
        where={"chunk_id": {"$in": chunk_ids}},
        include=["documents", "metadatas"],
    )

    if not results["documents"]:
        continue

    for document, metadata in zip(
        results["documents"],
        results["metadatas"],
    ):
        print("\n" + "=" * 100)
        print(f"STORE: {store_name}")
        print(f"SOURCE: {metadata.get('source')}")
        print(f"CHUNK ID: {metadata.get('chunk_id')}")
        print("=" * 100)
        print(document)