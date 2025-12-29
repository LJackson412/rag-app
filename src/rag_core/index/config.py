from __future__ import annotations

from typing import Annotated, Literal, TypeVar

from langchain_core.runnables import RunnableConfig, ensure_config
from pydantic import BaseModel, Field, PrivateAttr

from rag_core.index.prompts import GEN_METADATA_PROMPT
from rag_core.providers.factory import DefaultProviderFactory, ProviderFactory

T = TypeVar("T", bound="IndexConfig")


class IndexConfig(BaseModel):
    """Configurable Indexing Mode for RAG Index Graph."""
    embedding_model: Annotated[
        str,
        Field(
            default="openai/text-embedding-3-large",
            description=(
                "Embedding model used for indexing "
                "Use 'text-embedding-3-large' when you need maximum retrieval quality."
                "Use the same embedding model for Retrieval"
            ),
            json_schema_extra={
                "langgraph_nodes": ["save"],
            },
        ),
    ]
    mode: Annotated[
        Literal["none", "all", "imgs", "tables", "texts"],
        Field(
            default="all",
            description=(
                "Use 'imgs' to enrich only image segments while other content is saved without LLM enrichment."
            ),
        ),
    ]
    # ------------------------------------------------------------------------

    gen_metadata_model: Annotated[
        str,
        Field(
            default="openai/gpt-4.1",
            description=(
                "Model for metadata enrichment across text, image and table segments "
                "Generates metadata for better retrieval quality "
                "As an example, a retrieval_summary is formed, which is used to retrieve this segment"
            ),
            json_schema_extra={
                "langgraph_nodes": [
                    "extract_text",
                    "extract_imgs",
                    "extract_tables",
                ],
            },
        ),
    ]
    gen_metadata_prompt: str = Field(
        default=GEN_METADATA_PROMPT,
        description="System prompt for generating metadata",
        json_schema_extra={
            "langgraph_nodes": ["extract_text"],
            "langgraph_type": "prompt",
        },
    )
    vstore: Annotated[
        str,
        Field(
            default="chroma",
            description="Vector store provider",
            json_schema_extra={
                "langgraph_nodes": ["save"],
            },
        ),
    ]
    # ------------------------------------------------------------------------
    _provider_factory: ProviderFactory = PrivateAttr(default_factory=DefaultProviderFactory)
    
    @property
    def provider_factory(self) -> ProviderFactory:
        return self._provider_factory

    @classmethod
    def from_runnable_config(cls: type[T], config: RunnableConfig | None = None) -> T:
        """Create an IndexConfiguration instance from a RunnableConfig object."""
        config = ensure_config(config)
        configurable = config.get("configurable", {}) or {}
        valid_fields = set(cls.model_fields.keys())
        filtered = {k: v for k, v in configurable.items() if k in valid_fields}
        return cls(**filtered)
