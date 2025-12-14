from __future__ import annotations

from typing import Protocol

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.vectorstores import VectorStore

from rag_app.providers.chat_model import get_chat_model
from rag_app.providers.embedding import get_embedding
from rag_app.providers.vstore import get_vstore

_DEFAULT_RATE_LIMITER = InMemoryRateLimiter()


class ProviderFactory(Protocol):
    def build_chat_model(
        self,
        provider: str = "openai",
        model_name: str = "gpt-4.1",
        temp: float = 0.0,
        max_retries: int = 5,
        rate_limiter: InMemoryRateLimiter | None = None,
    ) -> BaseChatModel: ...
    
    def build_embeddings(
        self,
        provider: str = "openai",
        model_name: str = "text-embedding-3-large"
    ) -> Embeddings: ...
    
    def build_vstore(
        self,
        embedding_model: Embeddings,
        provider: str= "chroma",
        collection_name: str = "chroma_collection",
        persist_directory: str = ".chroma_directory"
    ) -> VectorStore: ...



class DefaultProviderFactory:
    def build_chat_model(
        self,
        provider: str = "openai",
        model_name: str = "gpt-4.1",
        temp: float = 0.0,
        max_retries: int = 5,
        rate_limiter: InMemoryRateLimiter | None = None,
    ) -> BaseChatModel:
        return get_chat_model(
            provider=provider,
            model_name=model_name,
            temp=temp,
            max_retries=max_retries,
            rate_limiter=rate_limiter or _DEFAULT_RATE_LIMITER,
        )

    def build_embeddings(
        self,
        provider: str = "openai",
        model_name: str = "text-embedding-3-large"
    ) -> Embeddings:
        return get_embedding(
            provider=provider,
            model_name=model_name,
        )

    def build_vstore(
        self,
        embedding_model: Embeddings,
        provider: str = "chroma",
        collection_name: str = "chroma_collection",
        persist_directory: str = ".chroma_directory"
    ) -> VectorStore:

        return get_vstore(
            provider=provider,
            embedding_model=embedding_model,
            collection_name=collection_name,
            persist_directory=persist_directory,
        )


_DEFAULT_PROVIDER_FACTORY = DefaultProviderFactory()


def get_provider_factory(factory: ProviderFactory | None = None) -> ProviderFactory:
    return factory or _DEFAULT_PROVIDER_FACTORY
