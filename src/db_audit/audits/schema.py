from __future__ import annotations

from typing import Literal

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
    requirement: str
    judgment: Judgment
    statement: str
    chunk_ids: list[str] = Field(default_factory=list)

class LLMAuditResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sub_requirements: list[LLMSubRequirementAudit]
    overall_judgment: Judgment
    overall_statement: str


# --- Audit Result Schema Definitions ------------------------------------
class SubRequirementAudit(LLMSubRequirementAudit):
    evidence_docs: list[Document] = Field(default_factory=list)

class AuditResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sub_requirements: list[SubRequirementAudit]
    overall_judgment: Judgment
    overall_statement: str

    @classmethod
    def from_llm(cls, llm: LLMAuditResult) -> "AuditResult":
        return cls.model_validate(llm.model_dump())

    def attach_documents(self, docs: list[Document], id_key: str = "id") -> None:
        docs_by_id = {str(d.metadata.get(id_key)): d for d in docs if d.metadata.get(id_key) is not None}
        for sr in self.sub_requirements:
            sr.evidence_docs = [docs_by_id[cid] for cid in sr.chunk_ids if cid in docs_by_id]
