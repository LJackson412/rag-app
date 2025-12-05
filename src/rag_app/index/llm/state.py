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


class InputIndexState(BaseModel):
    path: Annotated[
        str,
        Field(
            ...,
            description=(
                "Absolute or repo-relative filesystem path to the PDF that should be indexed. "
                "The path is used by the extractor node to read the document before creating RAG chunks."
            ),
        ),
    ]


class OutputIndexState(BaseModel):
    text_segments: Annotated[
        list[TextSegment],
        Field(
            default_factory=list,
            description=(
                "Structured extraction output per PDF page returned by the extract node. Each entry contains the "
                "raw content, retrieval summaries, and metadata needed to build vector-store documents."
            ),
        ),
    ]
    image_segments: Annotated[
        list[ImageSegment],
        Field(
            default_factory=list,
            description=(
                "Structured extraction output per PDF page returned by the extract node. Each entry contains the "
                "raw content, retrieval summaries, and metadata needed to build vector-store documents."
            ),
        ),
    ]
    table_segments: Annotated[
        list[TableOrListSegment],
        Field(
            default_factory=list,
            description=(
                "Structured extraction output per PDF page returned by the extract node. Each entry contains the "
                "raw content, retrieval summaries, and metadata needed to build vector-store documents."
            ),
        ),
    ]
    code_or_formula_segments: Annotated[
        list[CodeOrFormulaSegment],
        Field(
            default_factory=list,
            description=(
                "Structured extraction output per PDF page returned by the extract node. Each entry contains the "
                "raw content, retrieval summaries, and metadata needed to build vector-store documents."
            ),
        ),
    ]
    other_segments: Annotated[
        list[OtherSegment],
        Field(
            default_factory=list,
            description=(
                "Structured extraction output per PDF page returned by the extract node. Each entry contains the "
                "raw content, retrieval summaries, and metadata needed to build vector-store documents."
            ),
        ),
    ]
    index_docs: Annotated[
        list[Document],
        Field(
            default_factory=list,
            description=(
                "LangChain Document objects persisted in Chroma during the save node. Includes chunk metadata such as "
                "doc_id, collection_id, and chunk_id that the retrieval graph relies on."
            ),
        ),
    ]


class OverallIndexState(InputIndexState, OutputIndexState):
    """Combined input/output schema used as the shared state across the graph."""

    document_metadata: Annotated[
        dict[str, Any],
        Field(
            default_factory=dict,
            description=("every page one text"),
        ),
    ]
    llm_errors: Annotated[
        list[Exception],
        add,
        Field(
            default_factory=list,
            description=(
                "Errors raised by the LLM structured extraction nodes. Clients can inspect these "
                "to selectively retry failed chunks without rerunning the entire indexing job."
            ),
        ),
    ]
