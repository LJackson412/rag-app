import asyncio

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from rag_core.retrieval.graph import graph as retrieval_graph
from rag_core.retrieval.prompts import (
    COMPRESS_DOCS_PROMPT,
    GENERATE_ANSWER_PROMPT,
    GENERATE_QUESTIONS_PROMPT,
)
from rag_core.retrieval.state import InputRetrievalState, OutputRetrievalState

"""
- Wrapper um rag_core/retrieval/graph.py
"""

async def retrieve_docs(
    collection_id: str,
    queries: list[str],
    generate_questions_prompt: str = GENERATE_QUESTIONS_PROMPT,
    compress_docs_prompt: str = COMPRESS_DOCS_PROMPT,
    generate_answer_prompt: str = GENERATE_ANSWER_PROMPT,
    number_of_llm_generated_questions: int = 3,
    include_original_question: bool = True,
    number_of_docs_to_retrieve: int = 10,
) -> list[OutputRetrievalState]:
    """
    - Dursuchen einer Collection nach relevanten Dokumenten basierend auf mehreren Abfragen
    """
    
    retrieval_config = RunnableConfig(
        configurable={
            "collection_id": collection_id,
            "generate_questions_prompt": generate_questions_prompt,
            "compress_docs_prompt": compress_docs_prompt,
            "generate_answer_prompt": generate_answer_prompt,
            "number_of_llm_generated_questions": number_of_llm_generated_questions,
            "include_original_question": include_original_question,
            "number_of_docs_to_retrieve": number_of_docs_to_retrieve
        },
    )
    
    retrieval_configs = []
    retrieval_states = []
    for query in queries:
        retrieval_state = InputRetrievalState(
            messages=[HumanMessage(content=query)]
        )
        
        retrieval_configs.append(retrieval_config)
        retrieval_states.append(retrieval_state)
    
    return await retrieval_graph.abatch(
        inputs=retrieval_states,
        config=retrieval_configs
    )


async def main() -> None:
    TEST_COLLECTION_ID = "Test"
    QUESTIONS = [
        "What is the main topic of the document?",
        "Summarize the key points discussed in the document."
    ]

    res = await retrieve_docs(TEST_COLLECTION_ID, QUESTIONS)

    print("Count Docs:", len(res))
    print("Type erstes Element:", type(res[0]))
    print("Erstes Element:", res[0])

if __name__ == "__main__":
    asyncio.run(main())