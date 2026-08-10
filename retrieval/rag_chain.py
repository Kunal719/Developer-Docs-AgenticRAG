from retrieval.base import BaseRetriever
from reranking.base_reranker import BaseReranker
from evaluation.observer import PipelineObserver
from langchain_core.language_models.chat_models import BaseChatModel
from prompt.rag_prompt import rag_prompt
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage
from utils.document_formatter import format_retrieved_context
from utils.helper import calculate_estimated_cost
from time import perf_counter

from config import GENERATION_INPUT_PRICE, GENERATION_OUTPUT_PRICE

def build_rag_chain(
        retriever: BaseRetriever, 
        reranker: BaseReranker, 
        llm: BaseChatModel,  
        prompt: ChatPromptTemplate = rag_prompt, 
        observer: PipelineObserver | None = None
        ):
    """
    Build the end-to-end Retrieval-Augmented Generation (RAG) chain.

    Args:
        retriever: Configured retriever used to fetch relevant documents.
        llm: Chat language model used for answer generation.
        prompt: Prompt template for the RAG pipeline.

    Returns:
        A runnable LCEL RAG chain.
    """

    def retrieve_and_rerank(query: str):
        """
        Retrieve documents and rerank them based on the query.
        """
        documents = retriever.retrieve(query, observer)
        reranked_documents = reranker.rerank(query, documents, observer)

        return {"question": query, "context": reranked_documents}

    # def retrieve_and_rerank(query: str):
    #     documents = retriever.retrieve(query)

    #     print("\n" + "=" * 80)
    #     print("BEFORE RERANK")
    #     print("=" * 80)

    #     for i, doc in enumerate(documents):
    #         print(f"\nDocument {i}")
    #         print(doc.metadata["source"])

    #         if "def attach_edge(" in doc.page_content:
    #             print("Contains")
    #         else:
    #             print("Not contains")

    #     reranked_documents = reranker.rerank(query, documents)

    #     print("\n" + "=" * 80)
    #     print("AFTER RERANK")
    #     print("=" * 80)

    #     for i, doc in enumerate(reranked_documents):
    #         print(f"\nDocument {i}")
    #         print(doc.metadata["source"])

    #         if "def attach_edge(" in doc.page_content:
    #             print("Contains")
    #         else:
    #             print("Not contains")

    #     return {
    #         "question": query,
    #         "context": reranked_documents,
    #     }

    retrieve_and_rerank_documents_runnable = RunnableLambda(retrieve_and_rerank)

    def format_documents(inputs):
        context = format_retrieved_context(inputs["context"])

        if observer is not None:
            observer.record_context(context)

        return {"question": inputs["question"], "context": context}

    format_documents_runnable = RunnableLambda(format_documents)

    def debug_prompt(messages):
        print("\n" + "=" * 100)
        print("PROMPT SENT TO LLM")
        print("=" * 100)

        for message in messages.messages:
            print(f"\n[{message.type.upper()}]")
            print(message.content)

        print("=" * 100 + "\n")

        return messages

    chain = (
        retrieve_and_rerank_documents_runnable
        | format_documents_runnable
        | prompt
        # | RunnableLambda(debug_prompt)
        | llm
    )

    return chain

def invoke_chain(chain, question: str, observer: PipelineObserver | None = None):

    start = perf_counter()
    response: AIMessage = chain.invoke(question)
    end = perf_counter()

    latency = end - start

    usage = response.usage_metadata

    if observer is not None and usage is not None:
        observer.record_generation(
            answer=response.content,
            answer_length=len(response.content),
            prompt_tokens=usage["input_tokens"],
            completion_tokens=usage["output_tokens"],
            total_tokens=usage["total_tokens"],
            estimated_cost=calculate_estimated_cost(
                input_tokens=usage["input_tokens"],
                output_tokens=usage["output_tokens"],
                input_price_per_million=GENERATION_INPUT_PRICE,
                output_price_per_million=GENERATION_OUTPUT_PRICE
            ),
            generation_latency=latency,
        )

    return response.content

def stream_chain(chain, question: str):
    for chunk in chain.stream(question):
        if chunk.content:
            yield chunk.content