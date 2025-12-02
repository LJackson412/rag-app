from typing import Any

from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph

from rag_app.factory.factory import abuild_vstore, build_chat_model
from rag_app.index.ocr.config import IndexConfig
from rag_app.index.ocr.mapping import map_to_docs
from rag_app.index.ocr.schema import DocumentSegment, Image, Text
from rag_app.index.ocr.state import (
    InputIndexState,
    OutputIndexState,
    OverallIndexState,
)
from rag_app.llm_enrichment.llm_enrichment import gen_llm_metadata
from rag_app.loader.loader import (
    load_imgs_from_pdf,
    load_pdf_metadata,
    load_texts_from_pdf,
)
from rag_app.utils.utils import make_chunk_id

splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""], chunk_size=900, add_start_index=True
)


async def extract_text(
    state: OverallIndexState,
    config: RunnableConfig,
) -> dict[str, Any]:

    index_config = IndexConfig.from_runnable_config(config)

    gen_metadata_prompt = index_config.gen_text_metadata_prompt
    gen_metadata_model = index_config.gen_metadata_model
    embedding_model = index_config.embedding_model
    doc_id = index_config.doc_id
    collection_id = index_config.collection_id


    base_metadata = load_pdf_metadata(state.path)
    pdf_texts = load_texts_from_pdf(state.path)

    chunks = []
    pages = []
    for pdf_text in pdf_texts:
        page_chunks = splitter.split_text(pdf_text.text)
        for chunk in page_chunks:
            chunks.append(chunk)
            pages.append(pdf_text.page_number)

    llm_chunks_metadata = await gen_llm_metadata(
        chunks,
        build_chat_model(gen_metadata_model),
        gen_metadata_prompt,
        Text
    )

    document_segments = []

    for chunk_index, (chunk, chunk_page, llm_text_metadata) in enumerate(
        zip(chunks, pages, llm_chunks_metadata, strict=True)
    ):
        chunk_id = make_chunk_id(
            chunk_type="text",
            collection_id=collection_id,
            doc_id=doc_id,
            chunk_index=chunk_index,
        )

        document_segment = DocumentSegment(
            extracted_content=chunk,
            metadata={
                **base_metadata,
                "chunk_type": "Text",
                "page_number": chunk_page, 
                "chunk_index": chunk_index,
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "collection_id": collection_id,
                "embedding_model": embedding_model,
                "gen_metadata_model": gen_metadata_model,
            },
            text=llm_text_metadata,
            image=None,
            table=None,
        )
        document_segments.append(document_segment)

    return {
        "texts": pdf_texts,
        "imgs" : None,
        "document_segments": document_segments,
    }

    
async def extract_imgs(
    state: OverallIndexState,
    config: RunnableConfig,
) -> dict[str, Any]:

    index_config = IndexConfig.from_runnable_config(config)

    gen_metadata_prompt = index_config.gen_img_metadata_prompt
    gen_metadata_model = index_config.gen_metadata_model
    embedding_model = index_config.embedding_model
    doc_id = index_config.doc_id
    collection_id = index_config.collection_id

    base_metadata = load_pdf_metadata(state.path)
    pdf_imgs = load_imgs_from_pdf(state.path)  # List[PDFImage]

    img_urls = [img.image_url for img in pdf_imgs]

    llm_chunks_metadata = await gen_llm_metadata(
        img_urls,
        build_chat_model(gen_metadata_model),
        gen_metadata_prompt,
        Image,   
        imgs=True
    )

    document_segments: list[DocumentSegment] = []

    for chunk_index, (img, img_url, llm_img_metadata) in enumerate(
        zip(pdf_imgs, img_urls, llm_chunks_metadata, strict=True)
    ):
        chunk_id = make_chunk_id(
            chunk_type="img",
            collection_id=collection_id,
            doc_id=doc_id,
            chunk_index=chunk_index,
        )

        document_segment = DocumentSegment(
            extracted_content=img_url,  # oder img.img_base64, je nach Design
            metadata={
                **base_metadata,
                "chunk_type": "Image",
                "page_number": img.page_number,
                "ext": img.ext,
                "chunk_index": chunk_index,
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "collection_id": collection_id,
                "embedding_model": embedding_model,
                "gen_metadata_model": gen_metadata_model,
            },
            text=None,
            image=llm_img_metadata,
            table=None,
        )
        document_segments.append(document_segment)

    return {
        "texts": None,
        "imgs": pdf_imgs,
        "document_segments": document_segments,
    }

    
async def save(
    state: OverallIndexState, config: RunnableConfig
) -> dict[str, list[Document]]:
    
    index_config = IndexConfig.from_runnable_config(config)
    
    collection_id = index_config.collection_id
    embedding_model = index_config.embedding_model

    vstore = await abuild_vstore(embedding_model, collection_id)

    docs = map_to_docs(state.document_segments)
    index_docs = filter_complex_metadata(docs)

    if index_docs:
        await vstore.aadd_documents(index_docs)

    return {"index_docs": index_docs}


def route_by_mode(
    state: OverallIndexState, 
    config: RunnableConfig
) -> list[str]:
    index_config = IndexConfig.from_runnable_config(config)

    if index_config.mode == "text":
        return ["extract_text"]
    elif index_config.mode == "images":
        return ["extract_imgs"]
    # elif index_config.mode == "tables":
    #      return ["index_tables"]
    elif index_config.mode == "all":
        return ["index_text", "index_images"]
    else:
        raise ValueError(f"Unsupported index mode: {index_config.mode}")



builder = StateGraph(
    state_schema=OverallIndexState,
    input_schema=InputIndexState,
    output_schema=OutputIndexState,
    context_schema=IndexConfig,
)


builder.add_node("extract_text", extract_text)
builder.add_node("extract_imgs", extract_imgs)
builder.add_node("save", save)

builder.add_conditional_edges(
    START,  
    route_by_mode,
    ["extract_imgs", "extract_text"],
)

builder.add_edge("extract_imgs", "save")
builder.add_edge("extract_text", "save")
# builder.add_edge("index_tables", "save")
builder.add_edge("save", END)

graph = builder.compile()
graph.name = "OCR-Indexer"