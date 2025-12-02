from __future__ import annotations

from typing import Annotated, Literal, TypeVar

from langchain_core.runnables import RunnableConfig, ensure_config
from pydantic import BaseModel, Field

from rag_app.prompts.prompts import GEN_IMG_METADATA_PROMPT, GEN_TEXT_METADATA_PROMPT

T = TypeVar("T", bound="IndexConfig")


class IndexConfig(BaseModel):
    """Configurable Indexing Mode for RAG Index Graph."""

    doc_id: Annotated[
        str,
        Field(
            description="document id for searching within a specific document",
            json_schema_extra={
                "langgraph_nodes": ["save"],
            },
        ),
    ]
    collection_id: Annotated[
        str,
        Field(
            description=(
                "ID of the collection to search within. "
                "A collection is a container of documents in the Chroma VStore. "
                "The value of collection_id must follow the required Chroma format."
            ),
            json_schema_extra={
                "langgraph_nodes": ["save"],
            },
        ),
    ]
    mode: Annotated[
        Literal["text", "images", "tables", "all"],
            Field(
            default="all",
            description=(
                ""
            ),
        )
    ]
    
    gen_metadata_model: Annotated[
        Literal["gpt-4o", "gpt-4o-mini"],
        Field(
            default="gpt-4o-mini",
            description=("Multimodal model for PDF extraction. "),
            json_schema_extra={
                "langgraph_nodes": ["extract"],
            },
        ),
    ]
    gen_text_metadata_prompt: str = Field(
        default=GEN_TEXT_METADATA_PROMPT,
        description="The system prompt used for generating responses.",
        json_schema_extra={
            "langgraph_nodes": ["extract_text"],
            "langgraph_type": "prompt",
        },
    )
    gen_img_metadata_prompt: str = Field(
        default=GEN_IMG_METADATA_PROMPT,
        description="The system prompt used for generating responses.",
        json_schema_extra={
            "langgraph_nodes": ["extract_imgs"],
            "langgraph_type": "prompt",
        },
    )
    embedding_model: Annotated[
        Literal["text-embedding-3-small", "text-embedding-3-large"],
        Field(
            default="text-embedding-3-small",
            description=(
                "OpenAI embedding model used for Chroma indexing. "
                "Use 'text-embedding-3-large' when you need maximum retrieval quality."
                "Use the same embedding model for Retrieval"
            ),
            json_schema_extra={
                "langgraph_nodes": ["save"],
            },
        ),
    ]

    @classmethod
    def from_runnable_config(cls: type[T], config: RunnableConfig | None = None) -> T:
        """Create an IndexConfiguration instance from a RunnableConfig object."""
        config = ensure_config(config)
        configurable = config.get("configurable", {}) or {}
        valid_fields = set(cls.model_fields.keys())
        filtered = {k: v for k, v in configurable.items() if k in valid_fields}
        return cls(**filtered)
