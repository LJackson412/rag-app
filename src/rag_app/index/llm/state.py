from operator import add
from typing import Annotated, Any

from langchain_core.documents import Document
from pydantic import BaseModel, Field

from rag_app.index.llm.schema import (
    CodeOrFormulaSegment,
    ImageSegment,
    OtherSegment,
    TableOrListSegment,
    TextSegment,
)
from rag_app.index.schema import LLMException


class InputIndexState(BaseModel):
    path: Annotated[
        str,
        Field(
            ...,
            description=(
                "Path to the PDF that should be indexed"
            ),
        ),
    ]


class OutputIndexState(BaseModel):
    text_segments: Annotated[
        list[TextSegment],
        Field(
            default_factory=list,
            description=(
                ""

            ),
        ),
    ]
    image_segments: Annotated[
        list[ImageSegment],
        Field(
            default_factory=list,
            description=(
                ""
            ),
        ),
    ]
    table_segments: Annotated[
        list[TableOrListSegment],
        Field(
            default_factory=list,
            description=(
                ""
            ),
        ),
    ]
    code_or_formula_segments: Annotated[
        list[CodeOrFormulaSegment],
        Field(
            default_factory=list,
            description=(
                ""
            ),
        ),
    ]
    other_segments: Annotated[
        list[OtherSegment],
        Field(
            default_factory=list,
            description=(
                ""
            ),
        ),
    ]
    index_docs: Annotated[
        list[Document],
        Field(
            default_factory=list,
            description=(
                ""
            ),
        ),
    ]
    llm_exceptions: Annotated[list[LLMException], add] = Field(
        default_factory=list,
        description=(
            "Errors raised by the LLM structured extraction nodes. Clients can inspect these "
            "to selectively retry failed chunks without rerunning the entire indexing job."
        ),
    )


class OverallIndexState(InputIndexState, OutputIndexState):
    """Combined input/output schema used as the shared state across the graph."""

    document_metadata: Annotated[
        dict[str, Any],
        Field(
            default_factory=dict,
            description=(""),
        ),
    ]
    
