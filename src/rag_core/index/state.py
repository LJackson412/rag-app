from typing import Annotated

from langchain_core.documents import Document
from pydantic import BaseModel, Field

from rag_core.loader.schema import Segment


class InputIndexState(BaseModel):
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
    path: Annotated[
        str,
        Field(
            description=("Path to the Dcoument that should be indexed"),
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
