import asyncio
from typing import Any

from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from rag_app.factory.factory import build_chat_model, build_vstore
from rag_app.index.llm.config import IndexConfig
from rag_app.index.llm.schema import LLMSegments, LLMTextSegment, TextSegment
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


async def extract_llm_structured_data(
    state: OverallIndexState, config: RunnableConfig
) -> dict[str, list[LLMSegments]]:
    index_config = IndexConfig.from_runnable_config(config)
    
    extract_model = index_config.extract_model
    extract_prompt = index_config.extract_data_prompt
    doc_id = index_config.doc_id
    collection_id = index_config.collection_id
    
    pdf_page_imgs = load_page_imgs_from_pdf(state.path)
    
    img_urls = [img.image_url for img in pdf_page_imgs]
    
    llm_segments = await gen_llm_structured_data_from_imgs(
        img_urls,
        build_chat_model(extract_model),
        extract_prompt,
        LLMSegments
    )
    
    # Mapping auf Segments
    text_segments: list[TextSegment] = []
    for chunk_index, (llm_segment, pdf_page_img) in enumerate(
        zip(llm_segments, pdf_page_imgs, strict=True)
    ):
        if isinstance(llm_segment, LLMTextSegment):
            chunk_id = make_chunk_id(
                chunk_type="Text",
                collection_id=collection_id,
                doc_id=doc_id,
                chunk_index=chunk_index,
            )

            segment = TextSegment(
                metadata={
                    **state.document_metadata,
                    "chunk_type": "Text",
                    "page_number": pdf_page_img.page_number,
                    "chunk_index": chunk_index,
                    "chunk_id": chunk_id,
                },
                llm_text_segment=llm_segment
            )
            
            text_segments.append(segment)



    return {"text_segments": text_segments}


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


builder = StateGraph(
    state_schema=OverallIndexState,
    input_schema=InputIndexState,
    output_schema=OutputIndexState,
    context_schema=IndexConfig,
)

builder.add_node("extract", extract)
builder.add_node("save", save)

builder.add_edge(START, "extract")
builder.add_edge("extract", "save")
builder.add_edge("save", END)

graph = builder.compile()
graph.name = "Indexer"
