import asyncio
from typing import Literal

from langchain_core.runnables import RunnableConfig

from rag_core.index.graph import graph as index_graph
from rag_core.index.state import InputIndexState, OutputIndexState

"""
- Wrapper um rag_core/index/graph.py
"""

async def index_files(
    collection_id: str,
    paths: list[str], 
    doc_ids: list[str],
    mode: Literal["none", "all", "imgs", "tables", "texts"] = "none",
) -> list[OutputIndexState]:
    """
    - Input sind Dokumente einer Collection
    - Jedes Dokument hate eine Doc ID 
    """
    
    index_configs = []
    index_states = []
    for path, doc_id in  zip(paths, doc_ids, strict=True):
        index_config = RunnableConfig(
            configurable={
                "collection_id": collection_id,
                "doc_id": doc_id,
                "mode" : mode
            },
        )
        index_state = InputIndexState(
            path=path
        )
        
        index_configs.append(index_config)
        index_states.append(index_state)
        
    
    results = await index_graph.abatch(
        inputs=index_states,
        config=index_configs
    )
    
    return [OutputIndexState.model_validate(r) for r in results]


async def main() -> None:
    TEST_COLLECTION_ID = "Test"
    TEST_PATH = ["./data/Test/Test.pdf"]
    TEST_DOC_ID = ["Test_PDF"]

    res = await index_files(TEST_COLLECTION_ID, TEST_PATH, TEST_DOC_ID)

    print("Count Docs:", len(res))
    print("Type erstes Element:", type(res[0]))
    print("Erstes Element:", res[0])

if __name__ == "__main__":
    asyncio.run(main())