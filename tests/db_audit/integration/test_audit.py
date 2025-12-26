from typing import Generator, TypedDict, cast

import pytest
from langchain_chroma import Chroma
from langchain_core.runnables import RunnableConfig

from db_audit.audits.config import AuditConfig
from db_audit.audits.graph import graph as audit_graph
from db_audit.audits.state import InputAuditState
from rag_core.index.config import IndexConfig
from rag_core.utils.utils import (
    extract_provider_and_model,
    get_provider_factory_from_config,
)

PATHS = [
    "./data/Test/Test.pdf",
    "./data/Test/Test.docx",
    "./data/Test/Test.csv",
    "./data/Test/Test.png",
    "./data/Test/Test.xlsx",
]

DOC_IDS = [
    "Test_pdf",
    "Test_docx",
    "Test_csv",
    "Test_png",
    "Test_xlsx",
]

REQUIREMENTS = [
    "Zugriffsberechtigungen von Benutzern unter Verantwortung des Unternehmens (interne und externe Mitarbeiter) werden mindestens jährlich überprüft, um diese zeitnah auf Änderungen im Beschäftigungsverhältnis (Kündigung, Versetzung, längerer Abwesenheit/Sabbatical/Elternzeit) anzupassen.",
    "Die Überprüfung erfolgt durch hierzu autorisierte Personen aus den Unternehmensbereichen des Unternehmens, die aufgrund ihres Wissens über die Zuständigkeiten die Angemessenheit der vergebenen Berechtigungen überprüfen können.",
    "Die Überprüfung sowie die sich daraus ergebenden Berechtigungsanpassungen werden nachvollziehbar dokumentiert.",
    "Administrative Berechtigungen werden regelmäßig (mind. jährlich) überprüft.",
]

WORKITEM_ELEMENT_ID = "Test_Audit_Case"


class AuditGraphData(TypedDict):
    config: RunnableConfig
    state: InputAuditState
    audit_config: AuditConfig


CASES = [
    {
        "input": {
            "workitem_element_id": WORKITEM_ELEMENT_ID,
            "mode": "all",
            "paths": PATHS,
            "doc_ids": DOC_IDS,
            "requirements": REQUIREMENTS,
        },
        "config": {},
    },
    
]

CASE_IDS = [
    "default-config",
]


@pytest.fixture(params=CASES, ids=CASE_IDS)
def create_audit_config_and_input(
    request: pytest.FixtureRequest,
) -> Generator[AuditGraphData, None, None]:
    case = request.param

    config = RunnableConfig(configurable=case.get("config", {}))
    audit_config = AuditConfig.from_runnable_config(config)
    state = InputAuditState(**case["input"])

    yield {
        "config": config,
        "state": state,
        "audit_config": audit_config,
    }

    index_config = RunnableConfig(
        configurable={
            "collection_id": WORKITEM_ELEMENT_ID,
            "doc_id": DOC_IDS[0],
        }
    )
    config = IndexConfig.from_runnable_config(index_config)
    provider_factory = get_provider_factory_from_config(index_config)

    embedding_provider, model_name = extract_provider_and_model(config.embedding_model)
    embedding_model = provider_factory.build_embeddings(
        provider=embedding_provider, model_name=model_name
    )

    vstore = cast(
        Chroma,
        provider_factory.build_vstore(embedding_model, config.vstore, config.collection_id),
    )
    vstore.delete_collection()


@pytest.mark.asyncio
async def test_audit_graph(create_audit_config_and_input: AuditGraphData) -> None:
    data = create_audit_config_and_input

    result = await audit_graph.ainvoke(
        input=data["state"],
        config=data["config"],
    )

    index_states = result.get("index_states") or []
    if data["audit_config"].skip_index:
        assert index_states == []
    else:
        assert len(index_states) > 0

    retrieval_states = result.get("retrieval_states") or []
    assert len(retrieval_states) > 0
    for retrieval_state in retrieval_states:
        assert retrieval_state.llm_answer is not None
