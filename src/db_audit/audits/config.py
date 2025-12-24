from __future__ import annotations

from typing import Annotated, Type, TypeVar

from langchain_core.runnables import RunnableConfig, ensure_config
from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema

from db_audit.audits.prompts import AUDIT_PROMPT
from db_audit.audits.schema import LLMAuditResult
from rag_core.retrieval.schema import LLMAnswer

T = TypeVar("T", bound="AuditConfig")


class AuditConfig(BaseModel):
    """Configurable Indexing Mode for RAG Index Graph."""
    # Index:
    # Audit:
    number_of_docs_to_retrieve: Annotated[
        int,
        Field(
            default=38,
            description=(
                "4x38"
            ),
            json_schema_extra={
                "langgraph_nodes": ["audit"],
            },
        ),
    ]
    audit_prompt: str = Field(
        default=AUDIT_PROMPT,
        description="The system prompt used for generating responses.",
        json_schema_extra={
            "langgraph_nodes": ["generate_answer"],
            "langgraph_type": "prompt",
        },
    )
    audit_schema: SkipJsonSchema[Type[BaseModel]] = Field(
        default=LLMAuditResult,
        description="",
        json_schema_extra={"langgraph_nodes": ["generate_answer"]},
    )
    skip_index: bool = Field(
        default=True,
        description="",
        json_schema_extra={"langgraph_nodes": ["generate_answer"]},
    )

    @classmethod
    def from_runnable_config(cls: type[T], config: RunnableConfig | None = None) -> T:
        """Create an RetrievalConfiguration instance from a RunnableConfig object."""
        config = ensure_config(config)
        configurable = config.get("configurable", {}) or {}
        valid_fields = set(cls.model_fields.keys())
        filtered = {k: v for k, v in configurable.items() if k in valid_fields}
        return cls(**filtered)
