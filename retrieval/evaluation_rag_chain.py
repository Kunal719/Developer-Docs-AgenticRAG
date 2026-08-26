from retrieval.base import BaseRetriever
from reranking.base_reranker import BaseReranker
from evaluation.observer import PipelineObserver
from langchain_core.language_models.chat_models import BaseChatModel
from prompt.generation_prompt import generation_prompt
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from utils.document_formatter import format_retrieved_context
from utils.helper import calculate_estimated_cost
from time import perf_counter

def build_rag_chain(
        retriever: BaseRetriever, 
        reranker: BaseReranker, 
        llm: BaseChatModel,  
        prompt: ChatPromptTemplate = generation_prompt, 
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

    retrieve_and_rerank_documents = RunnableLambda(retrieve_and_rerank)

    format_documents = RunnableLambda(
        lambda inputs: {
            "question": inputs["question"],
            "context": format_retrieved_context(inputs["context"])
        }
    )

    def invoke_llm(inputs):

        start_time = perf_counter()
        response = llm.invoke(inputs)
        end_time = perf_counter()

        latency = (end_time - start_time)

        usage = response.usage_metadata

        if observer is not None:
            observer.record_generation(
                answer=response.content,
                prompt_tokens=usage["input_tokens"],
                completion_tokens=usage["output_tokens"],
                total_tokens=usage["total_tokens"],
                estimated_cost=calculate_estimated_cost(
                    input_tokens=usage["input_tokens"], 
                    output_tokens=usage["output_tokens"]),
                generation_latency=latency,
            )

        return response

    chain = (
        retrieve_and_rerank_documents
        | format_documents
        | prompt
        | RunnableLambda(invoke_llm)
        | StrOutputParser()
    )

    return chain