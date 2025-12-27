from __future__ import annotations

from typing import Protocol

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.vectorstores import VectorStore

from rag_core.providers.chat_model import get_chat_model
from rag_core.providers.embedding import get_embedding
from rag_core.providers.vstore import get_vstore


class ProviderFactory(Protocol):
    def build_chat_model(
        self,
        provider: str,
        model_name: str,
        temp: float = 0.0,
        max_retries: int = 5,
        rate_limiter: InMemoryRateLimiter | None = None,
    ) -> BaseChatModel: ...
    
    def build_embeddings(
        self,
        provider: str,
        model_name: str 
    ) -> Embeddings: ...
    
    def build_vstore(
        self,
        embedding_model: Embeddings,
        provider: str,
        collection_name: str,
        persist_directory: str 
    ) -> VectorStore: ...


_DEFAULT_RATE_LIMITER = InMemoryRateLimiter()

class DefaultProviderFactory:
    def build_chat_model(
        self,
        provider: str,
        model_name: str,
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
        provider: str,
        model_name: str
    ) -> Embeddings:
        return get_embedding(
            provider=provider,
            model_name=model_name,
        )

    def build_vstore(
        self,
        embedding_model: Embeddings,
        provider: str,
        collection_name: str,
        persist_directory: str
    ) -> VectorStore:

        return get_vstore(
            provider=provider,
            embedding_model=embedding_model,
            collection_name=collection_name,
            persist_directory=persist_directory,
        )
