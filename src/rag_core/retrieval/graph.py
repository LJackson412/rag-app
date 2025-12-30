import asyncio
import logging
import time
from collections.abc import Sequence
from typing import Any, Dict, cast

from langchain_core.documents import Document
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from rag_core.retrieval.config import RetrievalConfig
from rag_core.retrieval.schema import LLMDecision, LLMQuestions
from rag_core.retrieval.state import (
    InputRetrievalState,
    OutputRetrievalState,
    OverallRetrievalState,
)
from rag_core.utils.utils import extract_provider_and_model

logger = logging.getLogger(__name__)

GRAPH_NAME = "Retriever"


def _log_ctx(state: Any, config: Any, node: str) -> dict[str, Any]:
    metadata = (config or {}).get("metadata", {}) or {}
    user_q = None
    try:
        user_q = getattr(state.messages[-1], "content", None)
    except Exception:
        pass

    return {
        "graph": GRAPH_NAME,
        "node": node,
        "collection_id": getattr(state, "collection_id", None),
        "doc_id": getattr(state, "doc_id", None),
        "run_id": metadata.get("run_id") or metadata.get("trace_id"),
        "user_question_len": len(user_q) if isinstance(user_q, str) else None,
    }


def _ms(start: float) -> int:
    return int((time.monotonic() - start) * 1000)


async def generate_questions(
    state: OverallRetrievalState, config: RunnableConfig
) -> dict[str, list[str]]:
    node = "generate_questions"
    ctx = _log_ctx(state, config, node)
    start = time.monotonic()

    retrieval_config = RetrievalConfig.from_runnable_config(config)
    provider_factory = retrieval_config.provider_factory
    
    generate_questions_provider, model_name = extract_provider_and_model(
        retrieval_config.generate_questions_model
    )

    number = retrieval_config.number_of_llm_generated_questions
    user_question = state.messages[-1].content

    logger.info(
        "Generate questions start",
        extra={
            **ctx,
            "provider": generate_questions_provider,
            "model": model_name,
            "number": number,
        },
    )

    structured_llm = provider_factory.build_chat_model(
        provider=generate_questions_provider,
        model_name=model_name,
        temp=0.3,  # for semantic relevant questions
    ).with_structured_output(LLMQuestions)
    
    generate_questions_prompt = retrieval_config.generate_questions_prompt

    prompt = PromptTemplate(
        input_variables=["question", "number"],
        template=generate_questions_prompt,
    )

    chain = prompt | structured_llm
    try:
        llm_output = cast(
            LLMQuestions,
            await chain.ainvoke({"question": user_question, "number": number}),
        )
    except Exception:
        logger.exception(
            "Generate questions failed",
            extra={
                **ctx,
                "provider": generate_questions_provider,
                "model": model_name,
            },
        )
        raise

    questions = llm_output.questions or []
    level = logging.WARNING if len(questions) == 0 else logging.INFO
    logger.log(
        level,
        "Generate questions done",
        extra={**ctx, "duration_ms": _ms(start), "generated_questions": len(questions)},
    )

    return {"llm_questions": questions}


async def retrieve_docs(
    state: OverallRetrievalState, config: RunnableConfig
) -> dict[str, list[Document]]:
    node = "retrieve"
    ctx = _log_ctx(state, config, node)
    start = time.monotonic()
    retrieval_config = RetrievalConfig.from_runnable_config(config)

    provider_factory = retrieval_config.provider_factory
    
    embedding_provider, model_name = extract_provider_and_model(
        retrieval_config.embedding_model
    )
    vstore_provider = retrieval_config.vstore

    k = retrieval_config.number_of_docs_to_retrieve
    include_original_question = retrieval_config.include_original_question
    user_question = cast(str, state.messages[-1].content)

    logger.info(
        "Retrieve start",
        extra={
            **ctx,
            "embedding_provider": embedding_provider,
            "embedding_model": model_name,
            "vstore_provider": vstore_provider,
            "k": k,
            "include_original_question": include_original_question,
            "filter_doc_id": state.doc_id is not None,
        },
    )

    embedding_model = provider_factory.build_embeddings(
        provider=embedding_provider, model_name=model_name
    )

    vstore_start = time.monotonic()
    try:
        vstore = await asyncio.to_thread(
            provider_factory.build_vstore,
            embedding_model,
            provider=vstore_provider,
            collection_name=state.collection_id,
            persist_directory=".chroma",
        )
    except Exception:
        logger.exception(
            "VStore build failed", extra={**ctx, "vstore_provider": vstore_provider}
        )
        raise
    logger.debug("VStore built", extra={**ctx, "duration_ms": _ms(vstore_start)})

    search_kwargs: Dict[str, Any] = {"k": k}

    if state.doc_id is not None:
        search_kwargs["filter"] = {"doc_id": state.doc_id}

    retriever = vstore.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs,
    )

    queries = (
        [user_question] + state.llm_questions
        if include_original_question
        else list(state.llm_questions)
    )
    logger.debug("Retriever batch", extra={**ctx, "queries": len(queries)})
    try:
        docs_per_query = await retriever.abatch(queries)
    except Exception:
        logger.exception("Retriever batch failed", extra={**ctx, "queries": len(queries)})
        raise

    all_docs = [doc for docs in docs_per_query for doc in docs]

    def _unique_documents(documents: Sequence[Document]) -> list[Document]:
        seen_contents = set()
        unique_docs = []

        for doc in documents:
            content = doc.metadata.get("id")
            if content not in seen_contents:
                seen_contents.add(content)
                unique_docs.append(doc)

        return unique_docs

    unique_docs = _unique_documents(all_docs)
    dropped = len(all_docs) - len(unique_docs)
    level = logging.WARNING if len(unique_docs) == 0 else logging.INFO
    logger.log(
        level,
        "Retrieve done",
        extra={
            **ctx,
            "duration_ms": _ms(start),
            "queries": len(queries),
            "retrieved_total": len(all_docs),
            "unique_docs": len(unique_docs),
            "dedup_dropped": dropped,
        },
    )

    return {"retrieved_docs": unique_docs}


async def compress_docs(
    state: OverallRetrievalState, config: RunnableConfig
) -> dict[str, list[Document]]:
    node = "compress_docs"
    ctx = _log_ctx(state, config, node)
    start = time.monotonic()
    retrieval_config = RetrievalConfig.from_runnable_config(config)

    provider_factory = retrieval_config.provider_factory

    compress_docs_provider, model_name = extract_provider_and_model(
        retrieval_config.compress_docs_model
    )

    compress_docs_prompt: str = retrieval_config.compress_docs_prompt
    user_question = state.messages[-1].content

    docs = list(state.retrieved_docs or [])
    cat_counts: dict[str, int] = {"Image": 0, "Table": 0, "Text": 0, "Other": 0}
    for doc in docs:
        category = doc.metadata.get("category")
        if category in cat_counts:
            cat_counts[category] += 1
        else:
            cat_counts["Other"] += 1

    logger.info(
        "Compress start",
        extra={
            **ctx,
            "provider": compress_docs_provider,
            "model": model_name,
            "retrieved_docs": len(docs),
            **{f"cat_{k}": v for k, v in cat_counts.items()},
        },
    )

    structured_llm = provider_factory.build_chat_model(
        provider=compress_docs_provider, model_name=model_name
    ).with_structured_output(LLMDecision)

    llm_inputs: list[LanguageModelInput] = []
    for doc in docs:
        
        if doc.metadata.get("category") == "Image":
            content = doc.metadata.get("img_url", "")
        elif doc.metadata.get("category") == "Table":
            content = doc.metadata.get("text_as_html") or doc.metadata.get("text", "")
        elif doc.metadata.get("category") == "Text":
            content = doc.metadata.get("text", "")
       

        if (
            doc.metadata.get("category") == "Image"
        ): 
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
                                "image_url": {"url": content},
                            },
                        ]
                    )
                ]
            )
        else:
            text_prompt = compress_docs_prompt.format(
                question=user_question,
                doc_content=content,
            )

            llm_inputs.append([HumanMessage(content=text_prompt)])

    try:
        llm_decisions = cast(list[LLMDecision], await structured_llm.abatch(llm_inputs))
    except Exception:
        logger.exception(
            "Compress batch failed",
            extra={
                **ctx,
                "provider": compress_docs_provider,
                "model": model_name,
                "inputs": len(llm_inputs),
            },
        )
        raise

    filtered_docs = [doc for dec, doc in zip(llm_decisions, docs, strict=True) if dec.is_relevant]
    relevant = len(filtered_docs)
    dropped = len(docs) - relevant

    level = logging.WARNING if relevant == 0 else logging.INFO
    logger.log(
        level,
        "Compress done",
        extra={
            **ctx,
            "duration_ms": _ms(start),
            "inputs": len(llm_inputs),
            "relevant": relevant,
            "dropped": dropped,
        },
    )

    return {"filtered_docs": filtered_docs}


async def generate_answer(
    state: OverallRetrievalState, config: RunnableConfig
) -> dict[str, BaseModel | list[Document]]:
    node = "generate_answer"
    ctx = _log_ctx(state, config, node)
    start = time.monotonic()

    retrieval_config = RetrievalConfig.from_runnable_config(config)

    provider_factory = retrieval_config.provider_factory

    generate_answer_provider, model_name = extract_provider_and_model(
        retrieval_config.generate_answer_model
    )
    generate_answer_prompt = retrieval_config.generate_answer_prompt
    user_question = cast(str, state.messages[-1].content)

    answer_schema = retrieval_config.generate_answer_schema
    structured_llm = provider_factory.build_chat_model(
        provider=generate_answer_provider, model_name=model_name
    ).with_structured_output(answer_schema)

    def _doc_text_for_prompt(doc: Document) -> str:
        cat = doc.metadata.get("category")
        if cat == "Image":
            return "Image"
        if cat == "Table":
            return doc.metadata.get("text_as_html") or doc.metadata.get("text", "") or ""
        if cat == "Text":
            return doc.metadata.get("text", "") or ""
        return doc.page_content or doc.metadata.get("text", "") or ""

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

        parts: list[str] = []
        for doc in docs:
            parts.append(
                chunk_tpl.format(
                    chunk_id=doc.metadata.get("id", "N/A"),
                    doc_content=_doc_text_for_prompt(doc),
                )
            )
        return "\n".join(parts)

    def _build_user_message(prompt: str, docs: list[Document]) -> HumanMessage:
        image_urls = []
        for doc in docs:
            if doc.metadata.get("category") == "Image":
                url = doc.metadata.get("img_url")
                if url:
                    image_urls.append(url)

        # no images -> plain text prompt only
        if not image_urls:
            return HumanMessage(content=prompt)

        # images exist -> multimodal
        content_parts: list[str | dict[str, Any]] = [{"type": "text", "text": prompt}]
        for url in image_urls:
            content_parts.append({"type": "image_url", "image_url": {"url": url}})

        return HumanMessage(content=content_parts)

    filtered_docs = list(state.filtered_docs or [])
    image_count = sum(1 for doc in filtered_docs if doc.metadata.get("category") == "Image")

    prompt = generate_answer_prompt.format(
        question=user_question,
        docs=_prepare_docs_for_prompt(filtered_docs),
    )

    llm_input = _build_user_message(prompt, filtered_docs)

    logger.info(
        "Generate answer start",
        extra={
            **ctx,
            "provider": generate_answer_provider,
            "model": model_name,
            "docs": len(filtered_docs),
            "images": image_count,
            "prompt_len": len(prompt),
        },
    )

    # TODO: reduce llm input to match model context size
    try:
        llm_answer = cast(BaseModel, await structured_llm.ainvoke([llm_input]))
    except Exception:
        logger.exception(
            "Generate answer failed",
            extra={**ctx, "provider": generate_answer_provider, "model": model_name},
        )
        raise

    chunk_ids = set(getattr(llm_answer, "chunk_ids", None) or [])
    llm_evidence_docs = [
        doc for doc in filtered_docs if doc.metadata.get("id") in chunk_ids
    ]

    level = logging.WARNING if (len(filtered_docs) > 0 and len(chunk_ids) == 0) else logging.INFO
    logger.log(
        level,
        "Generate answer done",
        extra={
            **ctx,
            "duration_ms": _ms(start),
            "chunk_ids": len(chunk_ids),
            "evidence_docs": len(llm_evidence_docs),
        },
    )

    return {"llm_answer": llm_answer, "llm_evidence_docs": llm_evidence_docs}



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
