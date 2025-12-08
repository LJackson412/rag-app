import asyncio
from collections.abc import Sequence
from typing import Any, Dict, cast

from langchain_core.documents import Document
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from rag_app.factory.factory import build_chat_model, build_vstore
from rag_app.retrieval.config import RetrievalConfig
from rag_app.retrieval.schema import LLMAnswer, LLMDecision, LLMQuestions
from rag_app.retrieval.state import (
    InputRetrievalState,
    OutputRetrievalState,
    OverallRetrievalState,
)


async def generate_questions(
    state: OverallRetrievalState, config: RunnableConfig
) -> dict[str, list[str]]:
    retrieval_config = RetrievalConfig.from_runnable_config(config)
    number = retrieval_config.number_of_llm_generated_questions
    generate_questions_model = retrieval_config.generate_questions_model
    user_question = state.messages[-1].content

    structured_llm = build_chat_model(
        model_name=generate_questions_model, temp=0.3  # for semantic relevant questions
    ).with_structured_output(LLMQuestions)

    generate_questions_prompt = retrieval_config.generate_questions_prompt

    prompt = PromptTemplate(
        input_variables=["question", "number"],
        template=generate_questions_prompt,
    )

    chain = prompt | structured_llm

    llm_output = cast(
        LLMQuestions,
        await chain.ainvoke({"question": user_question, "number": number}),
    )

    return {"llm_questions": llm_output.questions}


async def retrieve_docs(
    state: OverallRetrievalState, config: RunnableConfig
) -> dict[str, list[Document]]:
    retrieval_config = RetrievalConfig.from_runnable_config(config)
    doc_id = retrieval_config.doc_id
    collection_id = retrieval_config.collection_id
    embedding_model = retrieval_config.embedding_model
    k = retrieval_config.number_of_docs_to_retrieve
    include_original_question = retrieval_config.include_original_question
    user_question = cast(str, state.messages[-1].content)

    vstore = await asyncio.to_thread(build_vstore, embedding_model, collection_id)
    
    search_kwargs: Dict[str, Any] = {"k": k}

    if doc_id is not None:
        search_kwargs["filter"] = {"doc_id": doc_id}

    retriever = vstore.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs,
    )
  

    if include_original_question:
        queries = [user_question] + state.llm_questions
        docs_per_query = await retriever.abatch(queries)
    else:
        docs_per_query = await retriever.abatch(state.llm_questions)

    all_docs = [doc for docs in docs_per_query for doc in docs]

    def _unique_documents(documents: Sequence[Document]) -> list[Document]:
        seen_contents = set()
        unique_docs = []

        for doc in documents:
            content = doc.metadata.get("chunk_id", "N/A")
            if content not in seen_contents:
                seen_contents.add(content)
                unique_docs.append(doc)

        return unique_docs

    unique_docs = _unique_documents(all_docs)

    return {"retrieved_docs": unique_docs}


async def compress_docs(
    state: OverallRetrievalState, config: RunnableConfig
) -> dict[str, list[Document]]:
    retrieval_config = RetrievalConfig.from_runnable_config(config)
    compress_docs_model = retrieval_config.compress_docs_model
    compress_docs_prompt: str = retrieval_config.compress_docs_prompt  # dein String
    user_question = state.messages[-1].content

    structured_llm = build_chat_model(compress_docs_model).with_structured_output(
        LLMDecision
    )

    llm_inputs: list[LanguageModelInput] = []
    for doc in state.retrieved_docs:
        
        extracted = doc.metadata.get("extracted_content", "N/A")
        
        if doc.metadata.get("chunk_type") == "ImageOCR": # Images from LLM-Indexer excluded, because LLM extract text from the image not base64
            text_prompt = compress_docs_prompt.format(
                question=user_question,
                doc_content="Image",
            )

            llm_inputs.append(
                [
                    HumanMessage(
                        content=[
                            {"type": "text", "text": text_prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": extracted},
                            },
                        ]
                    )
                ]
            )
        else:
            text_prompt = compress_docs_prompt.format(
                question=user_question,
                doc_content=extracted,
            )

            llm_inputs.append([HumanMessage(content=text_prompt)])

    llm_decisions = cast(list[LLMDecision], await structured_llm.abatch(llm_inputs))

    filtered_docs = []
    for dec, doc in zip(llm_decisions, state.retrieved_docs, strict=True):
        if dec.is_relevant:
            filtered_docs.append(doc)

    return {"filtered_docs": filtered_docs}


async def generate_answer(
    state: OverallRetrievalState, config: RunnableConfig
) -> dict[str, LLMAnswer | list[Document] | AIMessage]:

    retrieval_config = RetrievalConfig.from_runnable_config(config)
    generate_answer_model = retrieval_config.generate_answer_model
    generate_answer_prompt = retrieval_config.generate_answer_prompt
    user_question = state.messages[-1].content

    prompt = PromptTemplate(
        input_variables=["question", "docs"], template=generate_answer_prompt
    )

    strucuterd_llm = build_chat_model(generate_answer_model).with_structured_output(
        LLMAnswer, include_raw=True
    )

    chain = prompt | strucuterd_llm

    def _prepare_docs_for_prompt(docs: list[Document]) -> str:
        if not docs:
            return "No Documents found"

        chunk_tpl = (
            "Document-Metadata:\n"
            "Chunk-ID: {chunk_id}\n"
            "Document Segment:\n"
            "{doc_content}\n"
            "------------------------------------------------------------\n"
        )

        parts = []
        for doc in docs:
            meta = cast(dict[str, Any], doc.metadata)
            parts.append(
                chunk_tpl.format(
                    chunk_id=meta.get("chunk_id", "N/A"),
                    doc_content=meta.get("extracted_content", "N/A"),
                )
            )

        return "\n".join(parts)

    filtered_docs = state.filtered_docs
    inputs = {
        "question": user_question,
        "docs": _prepare_docs_for_prompt(filtered_docs),
    }

    result = await chain.ainvoke(inputs)

    ai_message = result["raw"]
    llm_answer = cast(LLMAnswer, result["parsed"])

    chunk_ids = llm_answer.chunk_ids
    llm_evidence_docs = [
        doc for doc in filtered_docs if doc.metadata.get("chunk_id") in chunk_ids
    ]

    return {
        "llm_answer": llm_answer,
        "llm_evidence_docs": llm_evidence_docs,
        "messages": [ai_message],
    }


builder = StateGraph(
    state_schema=OverallRetrievalState,
    input_schema=InputRetrievalState,
    output_schema=OutputRetrievalState,
    context_schema=RetrievalConfig,
)

builder.add_node("generate_questions", generate_questions)
builder.add_node("retrieve", retrieve_docs)
builder.add_node("compress_docs", compress_docs)
builder.add_node("generate_answer", generate_answer)


builder.add_edge(START, "generate_questions")
builder.add_edge("generate_questions", "retrieve")
builder.add_edge("retrieve", "compress_docs")
builder.add_edge("compress_docs", "generate_answer")
builder.add_edge("generate_answer", END)

graph = builder.compile()
graph.name = "Retriever"
