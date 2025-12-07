from __future__ import annotations

from typing import Annotated, Literal, TypeVar

from langchain_core.runnables import RunnableConfig, ensure_config
from pydantic import BaseModel, Field

from rag_app.index.llm.prompts import EXTRACT_DATA_FROM_PDF_PROMPT

T = TypeVar("T", bound="IndexConfig")


class IndexConfig(BaseModel):
    """Configurable Indexing Mode for RAG Index Graph."""
    collection_id: Annotated[
        str,
        Field(
            description=(
                "A collection is a container of documents in the VStore"
            ),
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
        Literal["text-embedding-3-small", "text-embedding-3-large"],
        Field(
            default="text-embedding-3-large",
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
    extract_model: Annotated[
        Literal["gpt-4.1", "gpt-4.1-mini"],
        Field(
            default="gpt-4.1",
            description=(
                "Multimodal model for PDF extraction"
                "Extract and splits the page content and metadata from each PDF page as Image"
            ),
            json_schema_extra={
                "langgraph_nodes": ["extract"],
            },
        ),
    ]
    extract_data_prompt: str = Field(
        default=EXTRACT_DATA_FROM_PDF_PROMPT,
        description="System prompt for generating extraction and splitting",
        json_schema_extra={
            "langgraph_nodes": ["extract_node"],
            "langgraph_type": "prompt",
        },
    )


    @classmethod
    def from_runnable_config(cls: type[T], config: RunnableConfig | None = None) -> T:
        """Create an IndexConfiguration instance from a RunnableConfig object."""
        config = ensure_config(config)
        configurable = config.get("configurable", {}) or {}
        valid_fields = set(cls.model_fields.keys())
        filtered = {k: v for k, v in configurable.items() if k in valid_fields}
        return cls(**filtered)
