from __future__ import annotations

from typing import Annotated, Literal, TypeVar

from langchain_core.runnables import RunnableConfig, ensure_config
from pydantic import BaseModel, Field

from rag_app.index.ocr.prompts import GEN_METADATA_PROMPT

T = TypeVar("T", bound="IndexConfig")


class IndexConfig(BaseModel):
    """Configurable Indexing Mode for RAG Index Graph."""

    collection_id: Annotated[
        str,
        Field(
            description=("A collection is a container of documents in the VStore"),
            json_schema_extra={
                "langgraph_nodes": ["save"],
            },
        ),
    ]
    doc_id: Annotated[
        str,
        Field(
            description="The document ID is used to identify a document when it is retrieved",
            json_schema_extra={
                "langgraph_nodes": ["save"],
            },
        ),
    ]
    embedding_model: Annotated[
        Literal[
            "openai/text-embedding-3-small",
            "openai/text-embedding-3-large",
        ],
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
        Literal["openai/gpt-4.1", "openai/gpt-4.1-mini"],
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
    splitter_seperators: list[str] = Field(
        default=["\n\n", "\n", " ", ""],
        description="Sepearatos for recursive text splitting",
        json_schema_extra={"langgraph_nodes": ["extract_text"]},
    )
    vstore: Annotated[
        Literal["chroma"],
        Field(
            default="chroma",
            description="Vector store provider",
            json_schema_extra={
                "langgraph_nodes": ["save"],
            },
        ),
    ]
    splitter_chunk_size: int = Field(
        default=900,
        description="Chunk Size for recursive text splitting",
        json_schema_extra={"langgraph_nodes": ["extract_text"]},
    )
    # ------------------------------------------------------------------------

    @classmethod
    def from_runnable_config(cls: type[T], config: RunnableConfig | None = None) -> T:
        """Create an IndexConfiguration instance from a RunnableConfig object."""
        config = ensure_config(config)
        configurable = config.get("configurable", {}) or {}
        valid_fields = set(cls.model_fields.keys())
        filtered = {k: v for k, v in configurable.items() if k in valid_fields}
        return cls(**filtered)
