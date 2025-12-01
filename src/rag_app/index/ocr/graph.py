from typing import Any

from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph

from rag_app.factory.factory import abuild_vstore, build_chat_model
from rag_app.index.ocr.config import IndexConfig
from rag_app.index.ocr.mapping import map_to_docs
from rag_app.index.ocr.schema import ExtractedData, Text
from rag_app.index.ocr.state import (
    InputIndexState,
    OutputIndexState,
    OverallIndexState,
)
from rag_app.llm_enrichment.llm_enrichment import gen_llm_metadata
from rag_app.loader.loader import load_pdf_metadata, load_texts_from_pdf
from rag_app.utils.utils import make_chunk_id

splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""], chunk_size=900, add_start_index=True
)


async def extract_text(
    state: OverallIndexState, config: RunnableConfig
) -> dict[str, Any]:

    index_config = IndexConfig.from_runnable_config(config)
    gen_metadata_prompt = index_config.extract_data_prompt
    gen_metadata_model = index_config.extract_model
    
    base_metadata = load_pdf_metadata(state.path)
    texts = load_texts_from_pdf(state.path)

    chunks: list[str] = []
    chunk_page_numbers: list[int] = []
    for page_number, text in enumerate(texts, start=1):
        page_chunks = splitter.split_text(text)
        chunks.extend(page_chunks)
        chunk_page_numbers.extend([page_number] * len(page_chunks))
    
    llm_texts_metadata = await gen_llm_metadata(
        chunks,
        build_chat_model(gen_metadata_model),
        gen_metadata_prompt,
        Text
    )
    
    extracted_data_objs = []
    for chunk_page_number, llm_text_metadata in zip(chunk_page_numbers, llm_texts_metadata):
        extracted_data = ExtractedData(
            text=llm_text_metadata,
            figure=None,
            table=None,
            metadata={**base_metadata, "page_number": chunk_page_number}
        )
        extracted_data_objs.append(extracted_data)

    return {
        "metadata": base_metadata,
        "texts": texts,
        "chunks" : chunks,
        "extracted_data" : extracted_data_objs
    }


async def save(
    state: OverallIndexState, config: RunnableConfig
) -> dict[str, list[Document]]:
    index_config = IndexConfig.from_runnable_config(config)
    doc_id = index_config.doc_id
    collection_id = index_config.collection_id
    embedding_model = index_config.embedding_model
    extract_model = index_config.extract_model

    vstore = await abuild_vstore(embedding_model, collection_id)

    docs = map_to_docs(state.extracted_data)
    index_docs = filter_complex_metadata(docs)

    for i, doc in enumerate(index_docs, start=0):
        chunk_id = make_chunk_id(
            collection_id=collection_id, doc_id=doc_id, chunk_index=i
        )
        doc.metadata = {
            **getattr(doc, "metadata", {}),
            "doc_id": doc_id,
            "collection_id": collection_id,
            "chunk_index": i,
            "chunk_id": chunk_id,
            "embedding_model": embedding_model,
            "extract_model": extract_model,
        }

    if index_docs:
        await vstore.aadd_documents(index_docs)

    return {"index_docs": index_docs}


builder = StateGraph(
    state_schema=OverallIndexState,
    input_schema=InputIndexState,
    output_schema=OutputIndexState,
    context_schema=IndexConfig,
)

builder.add_node("extract", extract_text)
builder.add_node("save", save)

builder.add_edge(START, "extract")
builder.add_edge("extract", "save")
builder.add_edge("save", END)

graph = builder.compile()
graph.name = "Indexer"