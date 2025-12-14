from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from rag_app.config.settings import settings


def get_openai_embedding(
    model_name: str
) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=model_name, openai_api_key=settings.API_KEY)

def get_embedding(provider: str = "openai", model_name: str = "text-embedding-3-large") -> Embeddings:
    match provider:
        case "openai":
            return get_openai_embedding(model_name)
        case _:
            raise ValueError(f"Unknown provider: {provider}")
