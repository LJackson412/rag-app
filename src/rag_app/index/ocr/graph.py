import asyncio
import traceback
from dataclasses import replace
from typing import Any

from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from rag_app.index.ocr.config import IndexConfig
from rag_app.index.ocr.loader import load
from rag_app.index.ocr.mapping import map_to_docs
from rag_app.index.ocr.schema import Segment
from rag_app.index.ocr.state import (
    InputIndexState,
    OutputIndexState,
    OverallIndexState,
)
from rag_app.index.schema import LLMException, LLMMetaData
from rag_app.llm_enrichment.llm_enrichment import (
    gen_llm_structured_data_from_imgs,
    gen_llm_structured_data_from_texts,
)
from rag_app.utils.utils import (
    extract_provider_and_model,
    get_provider_factory_from_config,
    make_chunk_id,
)


async def load_file(
    state: OverallIndexState,
    config: RunnableConfig,
) -> dict[str, Any]:

    index_config = IndexConfig.from_runnable_config(config)

    collection_id = index_config.collection_id
    doc_id = index_config.doc_id
    gen_metadata_model = index_config.gen_metadata_model
    embedding_model = index_config.embedding_model
    
    segs = await asyncio.to_thread(load, state.path)


    def add_metadata(segs: list[Segment]) -> list[Segment]:
        updated = []
        for i, s in enumerate(segs):
            chunk_id = make_chunk_id(
                chunk_type=s.category,
                collection_id=collection_id,
                doc_id=doc_id,
                chunk_index=i,
            )

            md = {
                **s.metadata,
                "doc_id": doc_id,
                "collection_id": collection_id,
                "embedding_model": embedding_model,
                "gen_metadata_model": gen_metadata_model
            }

            updated.append(replace(s, id=chunk_id, metadata=md))
        return updated

    tables = [s for s in segs if s.category == "Table"]
    imgs = [s for s in segs if s.category == "Image"]
    texts = [s for s in segs if s.category == "Text"]
    
    tables = add_metadata(tables)
    imgs   = add_metadata(imgs)
    texts  = add_metadata(texts)

    return {
        "tables": tables,
        "imgs": imgs,
        "texts": texts
    }


async def enrich_texts_with_llm(
    state: OverallIndexState,
    config: RunnableConfig,
) -> dict[str, Any]:
    index_config = IndexConfig.from_runnable_config(config)
    
    provider_factory = get_provider_factory_from_config(config)

    gen_metadata_prompt = index_config.gen_metadata_prompt
    gen_metadata_provider, model_name = extract_provider_and_model(
        index_config.gen_metadata_model
    )

    texts_segs = state.texts

    texts_as_string = [ts.text for ts in texts_segs]
    llm_resps = await gen_llm_structured_data_from_texts(
        texts_as_string,
        provider_factory.build_chat_model(
            provider=gen_metadata_provider, model_name=model_name
        ),
        gen_metadata_prompt,
        LLMMetaData,
    )

    enriched_texts = []
    for text_seg, llm_res in zip(texts_segs, llm_resps, strict=True):

        if isinstance(llm_res, Exception):
            llm_exception = LLMException(
                page_number=text_seg.page_number,
                message=str(llm_res),
                traceback="".join(traceback.format_exception(llm_res)),
            )
            et = replace(
                text_seg,
                llm_metadata=None,
                llm_exception=llm_exception
            )
        else:
            et = replace(
                text_seg,
                llm_metadata=llm_res,
                llm_exception=None
            )

        enriched_texts.append(et)

    return {
        "texts": enriched_texts,
    }


async def enrich_imgs_with_llm(
    state: OverallIndexState,
    config: RunnableConfig,
) -> dict[str, Any]:
    index_config = IndexConfig.from_runnable_config(config)
    
    provider_factory = get_provider_factory_from_config(config)

    gen_metadata_prompt = index_config.gen_metadata_prompt
    gen_metadata_provider, model_name = extract_provider_and_model(
        index_config.gen_metadata_model
    )
    img_segments = state.imgs

    img_urls = [img_seg.img_url for img_seg in img_segments]
    llm_resps = await gen_llm_structured_data_from_imgs(
        img_urls,
        provider_factory.build_chat_model(
            provider=gen_metadata_provider, model_name=model_name
        ),
        gen_metadata_prompt,
        LLMMetaData,
    )

    enriched_imgs = []
    for img_seg, llm_res in zip(img_segments, llm_resps, strict=True):

        if isinstance(llm_res, Exception):
            llm_exception = LLMException(
                page_number=img_seg.page_number,
                message=str(llm_res),
                traceback="".join(traceback.format_exception(llm_res)),
            )
            ei = replace(
                img_seg,
                llm_metadata=None,
                llm_exception=llm_exception
            )
        else:
            ei = replace(
                img_seg,
                llm_metadata=llm_res,
                llm_exception=None
            )

        enriched_imgs.append(ei)

    return {"imgs": enriched_imgs}


async def enrich_tables_with_llm(
    state: OverallIndexState,
    config: RunnableConfig,
) -> dict[str, Any]:
    index_config = IndexConfig.from_runnable_config(config)
    
    provider_factory = get_provider_factory_from_config(config)

    gen_metadata_prompt = index_config.gen_metadata_prompt
    gen_metadata_provider, model_name = extract_provider_and_model(
        index_config.gen_metadata_model
    )
    table_segments = state.tables

    tables_inputs = [(ts.text_as_html or ts.text) for ts in table_segments]
    llm_resps = await gen_llm_structured_data_from_texts(
        tables_inputs,
        provider_factory.build_chat_model(
            provider=gen_metadata_provider, model_name=model_name
        ),
        gen_metadata_prompt,
        LLMMetaData,
    )

    enriched_tables = []
    for ts, llm_res in zip(table_segments, llm_resps, strict=True):
        
        if isinstance(llm_res, Exception):
            llm_exception = LLMException(
                page_number=ts.page_number,
                message=str(llm_res),
                traceback="".join(traceback.format_exception(llm_res)),
            )
            et = replace(
                ts,
                llm_metadata=None,
                llm_exception=llm_exception
            )
        else:
            et = replace(
                ts,
                llm_metadata=llm_res,
                llm_exception=None
            )

        enriched_tables.append(et)

    return {"tables": enriched_tables}


async def save(state: OverallIndexState, config: RunnableConfig) -> dict[str, Any]:
    index_config = IndexConfig.from_runnable_config(config)
    
    provider_factory = get_provider_factory_from_config(config)

    collection_name = index_config.collection_id
    embedding_provider, model_name = extract_provider_and_model(
        index_config.embedding_model
    )
    vstore_provider = index_config.vstore

    embedding_model = provider_factory.build_embeddings(
        provider=embedding_provider, model_name=model_name
    )
    vstore = await asyncio.to_thread(
        provider_factory.build_vstore,
        embedding_model,
        provider=vstore_provider,
        collection_name=collection_name,
    )

    segments = state.texts + state.imgs + state.tables

    docs = map_to_docs(segments)

    index_docs = filter_complex_metadata(docs)

    if index_docs:
        await vstore.aadd_documents(index_docs)

    return {"index_docs": index_docs}


def route_by_mode(state: OverallIndexState, config: RunnableConfig) -> list[str]:
    index_config = IndexConfig.from_runnable_config(config)

    if index_config.mode == "none":
        return ["save"]
    if index_config.mode == "imgs":
        return ["enrich_imgs_with_llm"]
    if index_config.mode == "tables":
        return ["enrich_tables_with_llm"]
    if index_config.mode == "texts":
        return ["enrich_texts_with_llm"]
    elif index_config.mode == "all":
        return [
            "enrich_texts_with_llm",
            "enrich_imgs_with_llm",
            "enrich_tables_with_llm",
        ]
    else:
        raise ValueError(f"Unsupported index mode: {index_config.mode}")


builder = StateGraph(
    state_schema=OverallIndexState,
    input_schema=InputIndexState,
    output_schema=OutputIndexState,
    context_schema=IndexConfig,
)


builder.add_node("load", load_file)
builder.add_node("enrich_imgs_with_llm", enrich_imgs_with_llm)
builder.add_node("enrich_tables_with_llm", enrich_tables_with_llm)
builder.add_node("enrich_texts_with_llm", enrich_texts_with_llm)
builder.add_node("save", save)

builder.add_edge(START, "load")

builder.add_conditional_edges(
    "load",
    route_by_mode,
    ["enrich_imgs_with_llm", "enrich_tables_with_llm", "enrich_texts_with_llm", "save"],
)

builder.add_edge("enrich_texts_with_llm", "save")
builder.add_edge("enrich_imgs_with_llm", "save")
builder.add_edge("enrich_tables_with_llm", "save")
builder.add_edge("save", END)


graph = builder.compile()
graph.name = "OCR-Indexer"
