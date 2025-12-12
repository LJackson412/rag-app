from typing import Any, Generator, TypedDict

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from rag_app.factory.factory import build_vstore
from rag_app.index.ocr.config import IndexConfig
from rag_app.index.ocr.graph import graph as index_graph
from rag_app.index.ocr.state import InputIndexState
from rag_app.retrieval.graph import graph as retrieval_graph
from rag_app.retrieval.state import InputRetrievalState

QUESTION = "Was ist der Maßnahmenplan?"
DOC_PATH = "./data/Test_M/Test_M_1.pdf"


class IndexGraphData(TypedDict):
    index_config: RunnableConfig
    index_state: InputIndexState
    retrieval_config: RunnableConfig
    retrieval_state: InputRetrievalState


CASES = [
    [
        # Indexing config
        {
            "doc_id": "1_Test_M_1",
            "collection_id": "Test_M",
        },
        # Retrieval config
        {
            "doc_id": "1_Test_M_1",
            "collection_id": "Test_M",
        },
    ],
    [
        # Indexing config
        {
            "doc_id": "2_Test_M_1",
            "collection_id": "Test_M",
            "mode" : "none"
        },
        # Retrieval config
        {
            "doc_id": "2_Test_M_1",
            "collection_id": "Test_M",
        },
    ],

]


@pytest.fixture
def create_config_and_input(case: list[dict[str, Any]]) -> Generator[IndexGraphData, None, None]:
    index_config = RunnableConfig(
        configurable=case[0]
    )
    index_state = InputIndexState(path=DOC_PATH)

    retrieval_config = RunnableConfig(
        configurable=case[1]
    )
    retrieval_state = InputRetrievalState(
        messages=[HumanMessage(content=QUESTION)]
    )

    yield {
        "index_config": index_config,
        "index_state": index_state,
        "retrieval_config": retrieval_config,
        "retrieval_state": retrieval_state,
    }

    # cleanup pro Case
    config = IndexConfig.from_runnable_config(index_config)
    vstore = build_vstore(config.embedding_model, config.collection_id)
    vstore.delete_collection()


@pytest.mark.asyncio
@pytest.mark.parametrize("case", CASES, ids=["default", "mode-none"])
async def test_index_graph(create_config_and_input: IndexGraphData) -> None:
    data = create_config_and_input

    index_res = await index_graph.ainvoke(
        input=data["index_state"],
        config=data["index_config"],
    )
    assert len(index_res["index_docs"]) > 0

    retrieval_res = await retrieval_graph.ainvoke(
        input=data["retrieval_state"],
        config=data["retrieval_config"],
    )
    assert retrieval_res["llm_answer"] is not None
    assert retrieval_res["llm_evidence_docs"] is not None
