from typing import Generator, TypedDict

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from rag_app.factory.factory import build_vstore
from rag_app.index.ocr.config import IndexConfig
from rag_app.index.ocr.graph import graph as index_graph
from rag_app.index.ocr.state import InputIndexState
from rag_app.retrieval.graph import graph as retrieval_graph
from rag_app.retrieval.state import InputRetrievalState


class IndexGraphData(TypedDict):
    index_config: RunnableConfig
    index_state: InputIndexState
    retrieval_config: RunnableConfig
    retrieval_state: InputRetrievalState

@pytest.fixture
def create_config_and_input() -> Generator[IndexGraphData, None, None]:
    index_config = RunnableConfig(
        configurable={
            "doc_id": "Test_M_1",
            "collection_id": "Test_M",
        }
    )
    index_state = InputIndexState(path="./data/Test_M/Test_M_1.pdf")
    
    retrieval_config = RunnableConfig(
        configurable={
            "doc_id": "Test_M_1",
            "collection_id": "Test_M",
        }
    )
    retrieval_state = InputRetrievalState(
        messages=[HumanMessage(content="Was ist der Maßnahmenplan?")] 
    )

    yield {
        "index_config": index_config,
        "index_state": index_state,
        "retrieval_config": retrieval_config,
        "retrieval_state": retrieval_state,
    }
    config = IndexConfig.from_runnable_config(index_config)
    vstore = build_vstore(config.embedding_model, config.collection_id)
    vstore.delete_collection()


@pytest.mark.asyncio
async def test_index_graph(create_config_and_input: IndexGraphData) -> None:
    data = create_config_and_input

    input_index_state = data["index_state"]
    index_config = data["index_config"]

    index_res = await index_graph.ainvoke(
        input=input_index_state,
        config=index_config,
    )

    index_docs = index_res["index_docs"]
    assert len(index_docs) > 0
    
    
    input_retrieval_state = data["retrieval_state"]
    retrieval_config = data["retrieval_config"]
    
    retrieval_res = await retrieval_graph.ainvoke(
        input=input_retrieval_state,
        config=retrieval_config,
    )
    
    llm_answer = retrieval_res["llm_answer"]
    llm_evidence_docs = retrieval_res["llm_evidence_docs"]
    assert llm_answer is not None
    assert llm_evidence_docs is not None
    
