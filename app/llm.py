from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel

def get_llm(model_name: str, model_provider:str, max_tokens: int | None = None, temperature: float | None = None) -> BaseChatModel:
    """
    Return the configured chat language model to be used by the RAG pipeline. Temperature is 0.1

    Args:
        model_name: Name of the model
        model_provider: Provider of the chat model
    """
    return init_chat_model(model=model_name, model_provider=model_provider, max_tokens=max_tokens, temperature=temperature)