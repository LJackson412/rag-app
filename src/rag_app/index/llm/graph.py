import asyncio
from typing import Any

from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from rag_app.factory.factory import build_chat_model, build_vstore
from rag_app.index.llm.config import IndexConfig
from rag_app.index.llm.mapping import map_to_docs
from rag_app.index.llm.schema import (
    CodeOrFormulaSegment,
    ImageSegment,
    LLMSegments,
    OtherSegment,
    TableOrListSegment,
    TextSegment,
)
from rag_app.index.llm.state import (
    InputIndexState,
    OutputIndexState,
    OverallIndexState,
)
from rag_app.llm_enrichment.llm_enrichment import gen_llm_structured_data_from_imgs
from rag_app.loader.loader import load_page_imgs_from_pdf, load_pdf_metadata
from rag_app.utils.utils import make_chunk_id


async def extract_metadata(
    state: OverallIndexState,
    config: RunnableConfig,
) -> dict[str, Any]:

    index_config = IndexConfig.from_runnable_config(config)

    extract_model = index_config.extract_model
    embedding_model = index_config.embedding_model
    doc_id = index_config.doc_id
    collection_id = index_config.collection_id

    base_metadata = load_pdf_metadata(state.path)

    metadata = {
        **base_metadata,
        "doc_id": doc_id,
        "collection_id": collection_id,
        "embedding_model": embedding_model,
        "extract_model": extract_model,
    }

    return {"document_metadata": metadata}


async def llm_extract(
    state: OverallIndexState, config: RunnableConfig
) -> dict[str, Any]:
    index_config = IndexConfig.from_runnable_config(config)
    
    extract_model = index_config.extract_model
    extract_prompt = index_config.extract_data_prompt
    doc_id = index_config.doc_id
    collection_id = index_config.collection_id
    
    pdf_page_imgs = load_page_imgs_from_pdf(state.path)
    
    img_urls = [img.image_url for img in pdf_page_imgs]
    
    llm_segments: list[LLMSegments]
    extraction_errors: list[Exception]
    llm_segments, extraction_errors = await gen_llm_structured_data_from_imgs(
        img_urls,
        build_chat_model(extract_model),
        extract_prompt,
        LLMSegments,
    )
    
    text_segments: list[TextSegment] = []
    image_segments: list[ImageSegment] = []
    table_segments: list[TableOrListSegment] = []
    code_or_formula_segments: list[CodeOrFormulaSegment] = []
    other_segments: list[OtherSegment] = []

    chunk_index = 0
    for llm_segment, pdf_page_img in zip(
        llm_segments, pdf_page_imgs[: len(llm_segments)], strict=True
    ):
        for text_segment in llm_segment.texts:
            chunk_id = make_chunk_id(
                chunk_type="Text",
                collection_id=collection_id,
                doc_id=doc_id,
                chunk_index=chunk_index,
            )
            text_segments.append(
                TextSegment(
                    metadata={
                        **state.document_metadata,
                        "chunk_type": "Text",
                        "page_number": pdf_page_img.page_number,
                        "chunk_index": chunk_index,
                        "chunk_id": chunk_id,
                    },
                    llm_text_segment=text_segment,
                )
            )
            chunk_index += 1

        for image_segment in llm_segment.figures:
            chunk_id = make_chunk_id(
                chunk_type="Image",
                collection_id=collection_id,
                doc_id=doc_id,
                chunk_index=chunk_index,
            )
            image_segments.append(
                ImageSegment(
                    metadata={
                        **state.document_metadata,
                        "chunk_type": "Image",
                        "page_number": pdf_page_img.page_number,
                        "chunk_index": chunk_index,
                        "chunk_id": chunk_id,
                    },
                    llm_image_segment=image_segment,
                )
            )
            chunk_index += 1

        for table_segment in llm_segment.tables:
            chunk_id = make_chunk_id(
                chunk_type="Table",
                collection_id=collection_id,
                doc_id=doc_id,
                chunk_index=chunk_index,
            )
            table_segments.append(
                TableOrListSegment(
                    metadata={
                        **state.document_metadata,
                        "chunk_type": "Table",
                        "page_number": pdf_page_img.page_number,
                        "chunk_index": chunk_index,
                        "chunk_id": chunk_id,
                    },
                    llm_table_segment=table_segment,
                )
            )
            chunk_index += 1

        for code_or_formula in llm_segment.code_or_formulas:
            chunk_id = make_chunk_id(
                chunk_type="CodeOrFormula",
                collection_id=collection_id,
                doc_id=doc_id,
                chunk_index=chunk_index,
            )
            code_or_formula_segments.append(
                CodeOrFormulaSegment(
                    metadata={
                        **state.document_metadata,
                        "chunk_type": "CodeOrFormula",
                        "page_number": pdf_page_img.page_number,
                        "chunk_index": chunk_index,
                        "chunk_id": chunk_id,
                    },
                    llm_code_or_formula_segment=code_or_formula,
                )
            )
            chunk_index += 1

        for other_segment in llm_segment.others:
            chunk_id = make_chunk_id(
                chunk_type="Other",
                collection_id=collection_id,
                doc_id=doc_id,
                chunk_index=chunk_index,
            )
            other_segments.append(
                OtherSegment(
                    metadata={
                        **state.document_metadata,
                        "chunk_type": "Other",
                        "page_number": pdf_page_img.page_number,
                        "chunk_index": chunk_index,
                        "chunk_id": chunk_id,
                    },
                    llm_other_segment=other_segment,
                )
            )
            chunk_index += 1

    return {
        "text_segments": text_segments,
        "image_segments": image_segments,
        "table_segments": table_segments,
        "code_or_formula_segments": code_or_formula_segments,
        "other_segments": other_segments,
        # Surface LLM parsing errors so clients can decide whether to retry the failed chunks.
        "llm_errors": state.llm_errors + extraction_errors,
    }


async def save(
    state: OverallIndexState, config: RunnableConfig
) -> dict[str, list[Document]]:

    index_config = IndexConfig.from_runnable_config(config)

    collection_id = index_config.collection_id
    embedding_model = index_config.embedding_model

    vstore = await asyncio.to_thread(build_vstore, embedding_model, collection_id)

    segments = (
        state.text_segments
        + state.image_segments
        + state.table_segments
        + state.code_or_formula_segments
        + state.other_segments
    )
    docs = map_to_docs(segments)
    index_docs = filter_complex_metadata(docs)

    if index_docs:
        await vstore.aadd_documents(index_docs)

    return {"index_docs": index_docs}


builder = StateGraph(
    state_schema=OverallIndexState,
    input_schema=InputIndexState,
    output_schema=OutputIndexState,
    context_schema=IndexConfig,
)

builder.add_node("extract_metadata", extract_metadata)
builder.add_node("llm_extract", llm_extract)
builder.add_node("save", save)

builder.add_edge(START, "extract_metadata")
builder.add_edge("extract_metadata", "llm_extract")
builder.add_edge("llm_extract", "save")
builder.add_edge("save", END)

graph = builder.compile()
graph.name = "LLM-Indexer"
