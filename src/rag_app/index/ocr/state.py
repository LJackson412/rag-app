from typing import Annotated

from langchain_core.documents import Document
from pydantic import BaseModel, Field

from rag_app.index.ocr.schema import DocumentSegment


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
    texts: Annotated[
        list[str],
        Field(
            default_factory=list,
            description=(
                "every page one text"
            ),
        ),
    ]
    document_segments: Annotated[
        list[DocumentSegment],
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
