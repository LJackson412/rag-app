import asyncio
import traceback
from typing import Any

from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph

from rag_app.factory.factory import build_chat_model, build_vstore
from rag_app.index.ocr.config import IndexConfig
from rag_app.index.ocr.mapping import map_to_docs
from rag_app.index.ocr.schema import (
    ImageSegment,
    LLMImageSegment,
    LLMTableSegment,
    LLMTextSegment,
    TableSegment,
    TextSegment,
)
from rag_app.index.ocr.state import (
    InputIndexState,
    OutputIndexState,
    OverallIndexState,
)
from rag_app.index.schema import LLMException
from rag_app.llm_enrichment.llm_enrichment import (
    gen_llm_structured_data_from_imgs,
    gen_llm_structured_data_from_texts,
)
from rag_app.loader.loader import (
    load_imgs_from_pdf,
    load_pdf_metadata,
    load_tables_from_pdf,
    load_texts_from_pdf,
)
from rag_app.utils.utils import make_chunk_id


async def extract_metadata(
    state: OverallIndexState,
    config: RunnableConfig,
) -> dict[str, Any]:

    index_config = IndexConfig.from_runnable_config(config)

    gen_metadata_model = index_config.gen_metadata_model
    embedding_model = index_config.embedding_model
    doc_id = index_config.doc_id
    collection_id = index_config.collection_id

    base_metadata = load_pdf_metadata(state.path)

    metadata = {
        **base_metadata,
        "doc_id": doc_id,
        "collection_id": collection_id,
        "embedding_model": embedding_model,
        "gen_metadata_model": gen_metadata_model,
    }

    return {"document_metadata": metadata}


async def extract_text(
    state: OverallIndexState,
    config: RunnableConfig,
) -> dict[str, list[TextSegment]]:

    index_config = IndexConfig.from_runnable_config(config)

    gen_metadata_prompt = index_config.gen_text_metadata_prompt
    gen_metadata_model = index_config.gen_metadata_model
    separators = index_config.splitter_seperators
    chunk_size = index_config.splitter_chunk_size

    doc_id = index_config.doc_id
    collection_id = index_config.collection_id

    pdf_texts = load_texts_from_pdf(state.path)

    splitter = RecursiveCharacterTextSplitter(
        separators=separators, chunk_size=chunk_size
    )

    chunks = []
    pages = []
    for pdf_text in pdf_texts:
        page_chunks = splitter.split_text(pdf_text.text)
        for chunk in page_chunks:
            chunks.append(chunk)
            pages.append(pdf_text.page_number)

    llm_text_responses = await gen_llm_structured_data_from_texts(
        chunks,
        build_chat_model(gen_metadata_model),
        gen_metadata_prompt,
        LLMTextSegment,
    )

    document_segments = []
    text_exceptions: list[LLMException] = []
    for chunk_index, (chunk, chunk_page, llm_text_segment) in enumerate(
        zip(chunks, pages, llm_text_responses, strict=True)
    ):
        if isinstance(llm_text_segment, Exception):
            text_exceptions.append(
                LLMException(
                    page_number=chunk_page,
                    chunk_type="Text",
                    chunk_index=chunk_index,
                    message=str(llm_text_segment),
                    traceback="".join(
                        traceback.format_exception(llm_text_segment)
                    ),
                )
            )
            continue

        chunk_id = make_chunk_id(
            chunk_type="Text",
            collection_id=collection_id,
            doc_id=doc_id,
            chunk_index=chunk_index,
        )

        text_segment = TextSegment(
            extracted_content=chunk,
            metadata={
                **state.document_metadata,
                "chunk_type": "Text",
                "page_number": chunk_page,
                "chunk_index": chunk_index,
                "chunk_id": chunk_id,
            },
            llm_text_segment=llm_text_segment,
        )
        document_segments.append(text_segment)

    return {
        "text_segments": document_segments,
        "llm_exceptions": state.llm_exceptions + text_exceptions,
    }


async def extract_imgs(
    state: OverallIndexState,
    config: RunnableConfig,
) -> dict[str, list[ImageSegment]]:

    index_config = IndexConfig.from_runnable_config(config)

    gen_metadata_prompt = index_config.gen_img_metadata_prompt
    gen_metadata_model = index_config.gen_metadata_model
    doc_id = index_config.doc_id
    collection_id = index_config.collection_id

    pdf_imgs = load_imgs_from_pdf(state.path)

    img_urls = [img.image_url for img in pdf_imgs]
    llm_img_responses = await gen_llm_structured_data_from_imgs(
        img_urls,
        build_chat_model(gen_metadata_model),
        gen_metadata_prompt,
        LLMImageSegment,
    )

    document_segments = []
    image_exceptions: list[LLMException] = []
    for chunk_index, (img, img_url, llm_img_segment) in enumerate(
        zip(pdf_imgs, img_urls, llm_img_responses, strict=True)
    ):
        if isinstance(llm_img_segment, Exception):
            image_exceptions.append(
                LLMException(
                    page_number=img.page_number,
                    chunk_type="Image",
                    chunk_index=chunk_index,
                    message=str(llm_img_segment),
                    traceback="".join(
                        traceback.format_exception(llm_img_segment)
                    ),
                )
            )
            continue

        chunk_id = make_chunk_id(
            chunk_type="Image",
            collection_id=collection_id,
            doc_id=doc_id,
            chunk_index=chunk_index,
        )

        img_segment = ImageSegment(
            extracted_content=img_url,
            metadata={
                **state.document_metadata,
                "chunk_type": "Image",
                "page_number": img.page_number,
                "ext": img.ext,
                "chunk_index": chunk_index,
                "chunk_id": chunk_id,
            },
            llm_image_segment=llm_img_segment,
        )
        document_segments.append(img_segment)

    return {
        "image_segments": document_segments,
        "llm_exceptions": state.llm_exceptions + image_exceptions,
    }


async def extract_tables(
    state: OverallIndexState,
    config: RunnableConfig,
) -> dict[str, list[TableSegment]]:

    index_config = IndexConfig.from_runnable_config(config)

    gen_metadata_prompt = index_config.gen_table_metadata_prompt
    gen_metadata_model = index_config.gen_metadata_model
    doc_id = index_config.doc_id
    collection_id = index_config.collection_id

    pdf_tables = await asyncio.to_thread(load_tables_from_pdf, state.path)

    html_or_text_tables = []
    for t in pdf_tables:
        if t.html is not None:
            html_or_text_tables.append(t.html)
        elif t.text is not None:
            html_or_text_tables.append(t.text)
        else:
            html_or_text_tables.append("table extraction failed")

    llm_table_responses = await gen_llm_structured_data_from_texts(
        html_or_text_tables,
        build_chat_model(gen_metadata_model),
        gen_metadata_prompt,
        LLMTableSegment,
    )

    document_segments = []
    table_exceptions: list[LLMException] = []
    for chunk_index, (pdf_table, llm_table_segment) in enumerate(
        zip(pdf_tables, llm_table_responses, strict=True)
    ):
        if isinstance(llm_table_segment, Exception):
            table_exceptions.append(
                LLMException(
                    page_number=pdf_table.page_number,
                    chunk_type="Table",
                    chunk_index=chunk_index,
                    message=str(llm_table_segment),
                    traceback="".join(
                        traceback.format_exception(llm_table_segment)
                    ),
                )
            )
            continue

        chunk_id = make_chunk_id(
            chunk_type="Table",
            collection_id=collection_id,
            doc_id=doc_id,
            chunk_index=chunk_index,
        )

        table_segment = TableSegment(
            extracted_content=pdf_table.html,
            metadata={
                **state.document_metadata,
                "chunk_type": "Table",
                "page_number": pdf_table.page_number,
                "table_text": pdf_table.text,
                "chunk_index": chunk_index,
                "chunk_id": chunk_id,
            },
            llm_table_segment=llm_table_segment,
        )
        document_segments.append(table_segment)

    return {
        "table_segments": document_segments,
        "llm_exceptions": state.llm_exceptions + table_exceptions,
    }


async def save(
    state: OverallIndexState, config: RunnableConfig
) -> dict[str, list[Document]]:

    index_config = IndexConfig.from_runnable_config(config)

    collection_id = index_config.collection_id
    embedding_model = index_config.embedding_model

    vstore = await asyncio.to_thread(build_vstore, embedding_model, collection_id)

    segments = state.text_segments + state.image_segments + state.table_segments
    docs = map_to_docs(segments)
    index_docs = filter_complex_metadata(docs)

    if index_docs:
        await vstore.aadd_documents(index_docs)

    return {"index_docs": index_docs}


def route_by_mode(state: OverallIndexState, config: RunnableConfig) -> list[str]:
    index_config = IndexConfig.from_runnable_config(config)

    if index_config.mode == "text":
        return ["extract_text"]
    elif index_config.mode == "images":
        return ["extract_imgs"]
    elif index_config.mode == "tables":
        return ["extract_tables"]
    elif index_config.mode == "all":
        return ["extract_imgs", "extract_text", "extract_tables"]
    else:
        raise ValueError(f"Unsupported index mode: {index_config.mode}")


builder = StateGraph(
    state_schema=OverallIndexState,
    input_schema=InputIndexState,
    output_schema=OutputIndexState,
    context_schema=IndexConfig,
)


builder.add_node("extract_metadata", extract_metadata)
builder.add_node("extract_text", extract_text)
builder.add_node("extract_imgs", extract_imgs)
builder.add_node("extract_tables", extract_tables)
builder.add_node("save", save)

builder.add_edge(START, "extract_metadata")

builder.add_conditional_edges(
    "extract_metadata",
    route_by_mode,
    ["extract_imgs", "extract_text", "extract_tables"],
)

builder.add_edge("extract_imgs", "save")
builder.add_edge("extract_text", "save")
builder.add_edge("extract_tables", "save")
builder.add_edge("save", END)

graph = builder.compile()
graph.name = "OCR-Indexer"