from retrieval.base import BaseRetriever
from evaluation.observer import PipelineObserver
from langchain_core.language_models.chat_models import BaseChatModel
from prompt.generation_prompt import generation_prompt
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from utils.document_formatter import format_retrieved_context
from time import perf_counter
from utils.helper import calculate_estimated_cost

def build_hybrid_rag_chain(
        retriever: BaseRetriever, 
        llm: BaseChatModel, 
        prompt: ChatPromptTemplate = generation_prompt, 
        observer: PipelineObserver | None = None):
    """
    Build the end-to-end Retrieval-Augmented Generation (RAG) chain.

    Args:
        retriever: Configured retriever used to fetch relevant documents.
        llm: Chat language model used for answer generation.
        prompt: Prompt template for the RAG pipeline.

    Returns:
        A runnable LCEL RAG chain.
    """

    retrieve_documents = RunnableLambda(lambda query: retriever.retrieve(query, observer))
    format_documents = RunnableLambda(lambda docs: format_retrieved_context(docs))

    def invoke_llm(inputs):
        start_time = perf_counter()
        response = llm.invoke(inputs)
        end_time  = perf_counter()

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
        {
            "context":  retrieve_documents | format_documents,
            "question": RunnablePassthrough(),
        }
        | prompt
        | RunnableLambda(invoke_llm)
        | StrOutputParser()
    )

    return chain