from typing import Annotated

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from rag_core.index.state import OutputIndexState
from rag_core.retrieval.state import OutputRetrievalState

MessagesState = Annotated[list[AnyMessage], add_messages]


class InputAuditState(BaseModel):
    # Index:
    workitem_element_id: Annotated[
        str,
        Field(
            default="",
            description=(
                ""
            ),
            json_schema_extra={
                "langgraph_nodes": ["retrieve"],
            },
        ),
    ]

    paths: Annotated[
        list[str],
        Field(
            description=(""),
        ),
    ]
    doc_ids: Annotated[
        list[str],
        Field(
            description=(""),
        ),
    ]
    # Audit:
    requirements: Annotated[
        list[str],
        Field(
            description=("requirements for audits"),
        ),
    ]




class OutputAuditState(BaseModel):
    index_states: Annotated[
        list[OutputIndexState],
        Field(
            default_factory=list,
            description=("requirements for audits"),
        ),
    ]
    retrieval_states: Annotated[
        list[OutputRetrievalState],
        Field(
            default_factory=list,
            description=("requirements for audits"),
        ),
    ]
    audit_report_html: Annotated[
        str,
        Field(
            default="",
            description=("Rendered HTML report for the audit run"),
        ),
    ]


class OverallAuditState(InputAuditState, OutputAuditState):
    """"""
