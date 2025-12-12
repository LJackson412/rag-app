from typing import Annotated

from langchain_core.documents import Document
from pydantic import BaseModel, Field

from rag_app.index.ocr.schema import Segment


class InputIndexState(BaseModel):
    path: Annotated[
        str,
        Field(
            description=("Path to the PDF that should be indexed"),
        ),
    ]


class OutputIndexState(BaseModel):
    texts: Annotated[
        list[Segment],
        Field(
            default_factory=list,
            description=("text segements"),
        ),
    ]
    tables: Annotated[
        list[Segment],
        Field(
            default_factory=list,
            description=("table segements"),
        ),
    ]
    imgs: Annotated[
        list[Segment],
        Field(
            default_factory=list,
            description=("img segements"),
        ),
    ]
    index_docs: Annotated[
        list[Document],
        Field(
            default_factory=list,
            description=(
                "LangChain Document objects persisted in Chroma during the save node."
                "Segments are mapped to Document objects, before being stored in the vector database"
            ),
        ),
    ]


class OverallIndexState(InputIndexState, OutputIndexState):
    """Combined input/output schema used as the shared state across the graph."""
