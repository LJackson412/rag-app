from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

# Used for both Graphs 
class BaseLLMSegmentAttributes(BaseModel):
    language: Annotated[
        Literal["de", "eng", "n/a"],
        Field(
            description="Detected document segment language; all outputs must use this language.",
        ),
    ]
    title: Annotated[
        str,
        Field(
            description=(
                "Original title of this document section, if present. "
                "If the section has no explicit title, create a short, precise, and informative title "
                "that clearly describes the content of this section."
            ),
        ),
    ]
    retrieval_summary: Annotated[
        str,
        Field(
            description=(
                "Write a short, retrieval-oriented summary of this document section in the same language as the original text."
                "Capture the main topic, important entities and concepts, their relationships, and any crucial numbers, dates, or identifiers. "
                "Use key domain terms and phrases that users might search for. 1-3 sentences, no bullet points."
            ),
        ),
    ]
    labels: Annotated[
        list[str],
        Field(
            default_factory=list,
            description=(
                "List of 2-3 short tags to enrich this section with metadata, "
            ),
        ),
    ]


class LLMException(BaseModel):
    error: Annotated[str, Field(description="LLM extraction error message")]
    metadata: Annotated[
        dict[str, Any],
        Field(
            default_factory=dict,
            description="Contextual metadata for the failed chunk extraction.",
        ),
    ]
