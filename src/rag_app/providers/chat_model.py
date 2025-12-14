from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openai import ChatOpenAI

from rag_app.config.settings import settings


def get_openai_chat_model(
    model_name: str,
    temp: float = 0.0,
    max_retries: int = 5,
    rate_limiter: InMemoryRateLimiter | None = None,
) -> ChatOpenAI:
    return ChatOpenAI(
        api_key=settings.API_KEY,
        rate_limiter=rate_limiter,
        max_retries=max_retries,
        model=model_name,
        temperature=temp,
    )
    
def get_chat_model(
    provider: str = "openai", 
    model_name: str = "gpt-4.1",
    temp: float = 0.0,
    max_retries: int = 5,
    rate_limiter: InMemoryRateLimiter | None = None
) -> BaseChatModel:
    match provider:
        case "openai":
            return get_openai_chat_model(model_name, temp, max_retries, rate_limiter)
        case _:
            raise ValueError(f"Unknown provider: {provider}")
