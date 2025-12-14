from __future__ import annotations

from dataclasses import dataclass
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
    def build_chat_model(self) -> BaseChatModel: ...
    def build_embeddings(self) -> Embeddings: ...
    def build_vstore(self) -> VectorStore: ...


@dataclass(frozen=True)
class DefaultProviderFactory:
    # Chat
    chat_provider: str = "openai"
    chat_model_name: str = "gpt-4.1"
    chat_temp: float = 0.0
    chat_max_retries: int = 5
    rate_limiter: InMemoryRateLimiter = _DEFAULT_RATE_LIMITER

    # Embeddings
    embedding_provider: str = "openai"
    embedding_model_name: str = "text-embedding-3-large"

    # VStore
    vstore_provider: str = "chroma"
    collection_name: str = "chroma_collection"
    persist_directory: str = ".chroma_directory"

    def build_chat_model(self) -> BaseChatModel:
        return get_chat_model(
            provider=self.chat_provider,
            model_name=self.chat_model_name,
            temp=self.chat_temp,
            max_retries=self.chat_max_retries,
            rate_limiter=self.rate_limiter,
        )

    def build_embeddings(self) -> Embeddings:
        return get_embedding(
            provider=self.embedding_provider,
            model_name=self.embedding_model_name,
        )

    def build_vstore(self) -> VectorStore:
        embedding_model = self.build_embeddings()
        return get_vstore(
            provider=self.vstore_provider,
            embedding_model=embedding_model,
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
        )


_DEFAULT_PROVIDER_FACTORY = DefaultProviderFactory()


def get_provider_factory(factory: ProviderFactory | None = None) -> ProviderFactory:
    return factory or _DEFAULT_PROVIDER_FACTORY
