from __future__ import annotations

from typing import Annotated, Generic, Literal, TypeVar

from langchain_core.documents import Document
from pydantic import BaseModel, ConfigDict, Field

# ---LLM Audit Result Schema Definitions ------------------------------------
Judgment = Literal[
    "nicht_erfuellt",
    "teilweise_erfuellt",
    "vollstaendig_erfuellt",
    "nicht_beurteilbar",
]


class LLMSubRequirementAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requirement: Annotated[str, Field(description="Atomare, prüfbare Teilanforderung (Kurztext).")]
    judgment: Annotated[Judgment, Field(description="Urteil zur Teilanforderung.")]
    statement: Annotated[
        str,
        Field(description="Kurze Begründung. Bei nicht_beurteilbar MUSS enthalten: 'Keine Evidenz im Kontext gefunden.'"),
    ]
    chunk_ids: Annotated[list[str], Field(default_factory=list)]


TSub = TypeVar("TSub", bound=LLMSubRequirementAudit)


class LLMAuditResult(BaseModel, Generic[TSub]):
    model_config = ConfigDict(extra="forbid")

    sub_requirements: Annotated[list[TSub], Field(description="Liste der Audits pro Teilanforderung.")]
    overall_judgment: Annotated[Judgment, Field(description="Gesamturteil gemäß Aggregationslogik im Prompt.")]
    overall_statement: Annotated[str, Field(description="Kurze Gesamtbegründung.")]

# ---Extended Audit Result Schema Definitions --------------------------------

class SubRequirementAudit(LLMSubRequirementAudit):
    model_config = ConfigDict(extra="forbid")

    evidence_docs: Annotated[list[Document], Field(default_factory=list)]

    def attach_documents(self, docs_by_id: dict[str, Document], id_key: str = "id") -> None:
        self.evidence_docs = [
            docs_by_id[cid]
            for cid in self.chunk_ids
            if cid in docs_by_id
        ]


class AuditResult(LLMAuditResult[SubRequirementAudit]):
    model_config = ConfigDict(extra="forbid")

    def attach_documents(self, docs: list[Document], id_key: str = "id") -> None:
        docs_by_id = {
            str(d.metadata.get(id_key)): d
            for d in docs
            if d.metadata.get(id_key) is not None
        }
        for sr in self.sub_requirements:
            sr.attach_documents(docs_by_id, id_key=id_key)

LLMOnlyAuditResult = LLMAuditResult[LLMSubRequirementAudit]