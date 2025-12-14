import asyncio
from collections.abc import Sequence
from typing import Any, Dict, cast

from langchain_core.documents import Document
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from rag_app.retrieval.config import RetrievalConfig
from rag_app.retrieval.schema import LLMAnswer, LLMDecision, LLMQuestions
from rag_app.retrieval.state import (
    InputRetrievalState,
    OutputRetrievalState,
    OverallRetrievalState,
)
from rag_app.utils.utils import extract_provider_and_model, get_provider_factory_from_config


async def generate_questions(
    state: OverallRetrievalState, config: RunnableConfig
) -> dict[str, list[str]]:
    
    retrieval_config = RetrievalConfig.from_runnable_config(config)
    
    provider_factory = get_provider_factory_from_config(config)
    generate_questions_provider, model_name = extract_provider_and_model(
        retrieval_config.generate_questions_model
    )
     
    number = retrieval_config.number_of_llm_generated_questions
    user_question = state.messages[-1].content

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

    llm_output = cast(
        LLMQuestions,
        await chain.ainvoke({"question": user_question, "number": number}),
    )

    return {"llm_questions": llm_output.questions}


async def retrieve_docs(
    state: OverallRetrievalState, config: RunnableConfig
) -> dict[str, list[Document]]:
    retrieval_config = RetrievalConfig.from_runnable_config(config)
    
    provider_factory = get_provider_factory_from_config(config)
    
    embedding_provider, model_name = extract_provider_and_model(
        retrieval_config.embedding_model
    )
    vstore_provider = retrieval_config.vstore
    collection_name = retrieval_config.collection_id
    
    doc_id = retrieval_config.doc_id
    k = retrieval_config.number_of_docs_to_retrieve
    include_original_question = retrieval_config.include_original_question
    user_question = cast(str, state.messages[-1].content)
    
    embedding_model = provider_factory.build_embeddings(
        provider=embedding_provider, model_name=model_name
    )

    vstore = await asyncio.to_thread(
        provider_factory.build_vstore,
        embedding_model,
        provider=vstore_provider,
        collection_name=collection_name,
    )

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
            content = doc.metadata.get("id")
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

    provider_factory = get_provider_factory_from_config(config)

    compress_docs_provider, model_name = extract_provider_and_model(
        retrieval_config.compress_docs_model
    )
    
    compress_docs_prompt: str = retrieval_config.compress_docs_prompt
    user_question = state.messages[-1].content

    structured_llm = provider_factory.build_chat_model(
        provider=compress_docs_provider, model_name=model_name
    ).with_structured_output(LLMDecision)

    llm_inputs: list[LanguageModelInput] = []
    for doc in state.retrieved_docs:
        
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

    llm_decisions = cast(list[LLMDecision], await structured_llm.abatch(llm_inputs))

    filtered_docs = []
    for dec, doc in zip(llm_decisions, state.retrieved_docs, strict=True):
        if dec.is_relevant:
            filtered_docs.append(doc)

    return {"filtered_docs": filtered_docs}


async def generate_answer(
    state: OverallRetrievalState, config: RunnableConfig
) -> dict[str, LLMAnswer | list[Document]]:

    retrieval_config = RetrievalConfig.from_runnable_config(config)

    provider_factory = get_provider_factory_from_config(config)

    generate_answer_provider, model_name = extract_provider_and_model(
        retrieval_config.generate_answer_model
    )
    generate_answer_prompt = retrieval_config.generate_answer_prompt
    user_question = cast(str, state.messages[-1].content)

    structured_llm = provider_factory.build_chat_model(
        provider=generate_answer_provider, model_name=model_name
    ).with_structured_output(LLMAnswer)

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

    filtered_docs = state.filtered_docs

    prompt = generate_answer_prompt.format(
        question=user_question,
        docs=_prepare_docs_for_prompt(filtered_docs),
    )

    llm_input = _build_user_message(prompt, filtered_docs)

    # TODO: reduce llm input to match model context size
    llm_answer = cast(LLMAnswer, await structured_llm.ainvoke([llm_input]))

    chunk_ids = set(llm_answer.chunk_ids or [])
    llm_evidence_docs = [
        doc for doc in filtered_docs if doc.metadata.get("id") in chunk_ids
    ]

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
