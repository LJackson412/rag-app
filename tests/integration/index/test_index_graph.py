from typing import Any, Generator, TypedDict, cast

import pytest
from langchain_chroma import Chroma
from langchain_core.runnables import RunnableConfig

from rag_core.index.config import IndexConfig
from rag_core.index.graph import graph as index_graph
from rag_core.index.state import InputIndexState
from rag_core.utils.utils import extract_provider_and_model


class IndexGraphData(TypedDict):
    index_config: RunnableConfig
    index_state: InputIndexState


CASES = [
    [
        # Input State
        {
            "collection_id": "Test_Index_Graph",
            "doc_id": "Test_PDF",
            "path": "./data/Test/Test.pdf"
        },
        # Index Config
        {},
    ],
]


@pytest.fixture
def create_config_and_input(case: list[dict[str, Any]]) -> Generator[IndexGraphData, None, None]:
    run_index_config = RunnableConfig(
        configurable=case[1]
    )
    index_state = InputIndexState(**case[0])
    
    yield {
        "index_config": run_index_config,
        "index_state": index_state,
    }
    
    index_config = IndexConfig.from_runnable_config(run_index_config)
    
    provider_factory = index_config.provider_factory
    provider, model = extract_provider_and_model(index_config.embedding_model)
    
    embedding_model = provider_factory.build_embeddings(
        provider=provider,
        model_name=model
    )

    vstore = cast(Chroma, provider_factory.build_vstore(
        embedding_model=embedding_model,
        provider=provider,
        collection_name=index_state.collection_id,
        persist_directory=".chroma"
    ))
    
    vstore.delete_collection()


@pytest.mark.asyncio
@pytest.mark.parametrize("case", CASES,   ids=[
        "pdf-default",
    ],
)
async def test_index_graph(create_config_and_input: IndexGraphData) -> None:
    data = create_config_and_input

    index_res = await index_graph.ainvoke(
        input=data["index_state"],
        config=data["index_config"],
    )
    assert len(index_res["index_docs"]) > 0

    