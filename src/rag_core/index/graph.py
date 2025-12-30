import asyncio
import logging
import time
import traceback
from dataclasses import replace
from typing import Any

from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from rag_core.index.config import IndexConfig
from rag_core.index.mapping import map_to_docs
from rag_core.index.schema import LLMException, LLMMetaData
from rag_core.index.state import (
    InputIndexState,
    OutputIndexState,
    OverallIndexState,
)
from rag_core.llm_enrichment.llm_enrichment import (
    gen_llm_structured_data_from_imgs,
    gen_llm_structured_data_from_texts,
)
from rag_core.loader.loader import load
from rag_core.loader.schema import Segment
from rag_core.utils.utils import extract_provider_and_model, make_chunk_id

logger = logging.getLogger(__name__)


def _log_ctx(state: Any, config: Any, node: str) -> dict[str, Any]:
    metadata = (config or {}).get("metadata", {}) or {}
    return {
        "node": node,
        "doc_id": getattr(state, "doc_id", None),
        "collection_id": getattr(state, "collection_id", None),
        "path": getattr(state, "path", None),
        "run_id": metadata.get("run_id") or metadata.get("trace_id"),
    }


def _ms(start: float) -> int:
    return int((time.monotonic() - start) * 1000)


async def load_file(
    state: OverallIndexState,
    config: RunnableConfig,
) -> dict[str, Any]:

    node = "load"
    ctx = _log_ctx(state, config, node)
    start = time.monotonic()

    index_config = IndexConfig.from_runnable_config(config)

    gen_metadata_model = index_config.gen_metadata_model
    embedding_model = index_config.embedding_model

    logger.info(
        "Load start",
        extra={
            **ctx,
            "gen_metadata_model": gen_metadata_model,
            "embedding_model": embedding_model,
        },
    )

    segs = await asyncio.to_thread(load, state.path)


    def add_metadata(segs: list[Segment]) -> list[Segment]:
        updated = []
        for i, s in enumerate(segs):
            chunk_id = make_chunk_id(
                chunk_type=s.category,
                collection_id=state.collection_id,
                doc_id=state.doc_id,
                chunk_index=i,
            )

            md = {
                **s.metadata,
                "doc_id": state.doc_id,
                "collection_id": state.collection_id,
                "embedding_model": embedding_model,
                "gen_metadata_model": gen_metadata_model,
            }

            updated.append(replace(s, id=chunk_id, metadata=md))
        return updated

    tables = [s for s in segs if s.category == "Table"]
    imgs = [s for s in segs if s.category == "Image"]
    texts = [s for s in segs if s.category == "Text"]

    tables = add_metadata(tables)
    imgs   = add_metadata(imgs)
    texts  = add_metadata(texts)

    logger.info(
        "Load done",
        extra={
            **ctx,
            "duration_ms": _ms(start),
            "tables_count": len(tables),
            "imgs_count": len(imgs),
            "texts_count": len(texts),
            "total_segments": len(segs),
        },
    )

    return {
        "tables": tables,
        "imgs": imgs,
        "texts": texts
    }


async def enrich_texts_with_llm(
    state: OverallIndexState,
    config: RunnableConfig,
) -> dict[str, Any]:
    node = "enrich_texts_with_llm"
    ctx = _log_ctx(state, config, node)
    start = time.monotonic()
    index_config = IndexConfig.from_runnable_config(config)

    provider_factory = index_config.provider_factory

    gen_metadata_prompt = index_config.gen_metadata_prompt
    gen_metadata_provider, model_name = extract_provider_and_model(
        index_config.gen_metadata_model
    )

    texts_segs = state.texts

    texts_as_string = [ts.text for ts in texts_segs]
    logger.info(
        "LLM enrichment start",
        extra={
            **ctx,
            "kind": "texts",
            "items": len(texts_as_string),
            "provider": gen_metadata_provider,
            "model": model_name,
        },
    )
    llm_resps = await gen_llm_structured_data_from_texts(
        texts_as_string,
        provider_factory.build_chat_model(
            provider=gen_metadata_provider, model_name=model_name
        ),
        gen_metadata_prompt,
        LLMMetaData,
    )

    enriched_texts = []
    fail_count = 0
    for text_seg, llm_res in zip(texts_segs, llm_resps, strict=True):

        if isinstance(llm_res, Exception):
            fail_count += 1
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

    ok_count = len(enriched_texts) - fail_count
    level = logging.WARNING if fail_count else logging.INFO
    logger.log(
        level,
        "LLM enrichment done",
        extra={
            **ctx,
            "kind": "texts",
            "duration_ms": _ms(start),
            "items": len(enriched_texts),
            "ok": ok_count,
            "failed": fail_count,
            "provider": gen_metadata_provider,
            "model": model_name,
        },
    )

    return {
        "texts": enriched_texts,
    }


async def enrich_imgs_with_llm(
    state: OverallIndexState,
    config: RunnableConfig,
) -> dict[str, Any]:
    node = "enrich_imgs_with_llm"
    ctx = _log_ctx(state, config, node)
    start = time.monotonic()
    index_config = IndexConfig.from_runnable_config(config)

    provider_factory = index_config.provider_factory

    gen_metadata_prompt = index_config.gen_metadata_prompt
    gen_metadata_provider, model_name = extract_provider_and_model(
        index_config.gen_metadata_model
    )
    img_segments = state.imgs

    img_urls = [img_seg.img_url for img_seg in img_segments]
    logger.info(
        "LLM enrichment start",
        extra={
            **ctx,
            "kind": "imgs",
            "items": len(img_urls),
            "provider": gen_metadata_provider,
            "model": model_name,
        },
    )
    llm_resps = await gen_llm_structured_data_from_imgs(
        img_urls,
        provider_factory.build_chat_model(
            provider=gen_metadata_provider, model_name=model_name
        ),
        gen_metadata_prompt,
        LLMMetaData,
    )

    enriched_imgs = []
    fail_count = 0
    for img_seg, llm_res in zip(img_segments, llm_resps, strict=True):

        if isinstance(llm_res, Exception):
            fail_count += 1
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

    ok_count = len(enriched_imgs) - fail_count
    level = logging.WARNING if fail_count else logging.INFO
    logger.log(
        level,
        "LLM enrichment done",
        extra={
            **ctx,
            "kind": "imgs",
            "duration_ms": _ms(start),
            "items": len(enriched_imgs),
            "ok": ok_count,
            "failed": fail_count,
            "provider": gen_metadata_provider,
            "model": model_name,
        },
    )

    return {"imgs": enriched_imgs}


async def enrich_tables_with_llm(
    state: OverallIndexState,
    config: RunnableConfig,
) -> dict[str, Any]:
    node = "enrich_tables_with_llm"
    ctx = _log_ctx(state, config, node)
    start = time.monotonic()
    index_config = IndexConfig.from_runnable_config(config)

    provider_factory = index_config.provider_factory

    gen_metadata_prompt = index_config.gen_metadata_prompt
    gen_metadata_provider, model_name = extract_provider_and_model(
        index_config.gen_metadata_model
    )
    table_segments = state.tables

    tables_inputs = [(ts.text_as_html or ts.text) for ts in table_segments]
    logger.info(
        "LLM enrichment start",
        extra={
            **ctx,
            "kind": "tables",
            "items": len(tables_inputs),
            "provider": gen_metadata_provider,
            "model": model_name,
        },
    )
    llm_resps = await gen_llm_structured_data_from_texts(
        tables_inputs,
        provider_factory.build_chat_model(
            provider=gen_metadata_provider, model_name=model_name
        ),
        gen_metadata_prompt,
        LLMMetaData,
    )

    enriched_tables = []
    fail_count = 0
    for ts, llm_res in zip(table_segments, llm_resps, strict=True):

        if isinstance(llm_res, Exception):
            fail_count += 1
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

    ok_count = len(enriched_tables) - fail_count
    level = logging.WARNING if fail_count else logging.INFO
    logger.log(
        level,
        "LLM enrichment done",
        extra={
            **ctx,
            "kind": "tables",
            "duration_ms": _ms(start),
            "items": len(enriched_tables),
            "ok": ok_count,
            "failed": fail_count,
            "provider": gen_metadata_provider,
            "model": model_name,
        },
    )

    return {"tables": enriched_tables}


async def save(state: OverallIndexState, config: RunnableConfig) -> dict[str, Any]:
    node = "save"
    ctx = _log_ctx(state, config, node)
    start = time.monotonic()
    index_config = IndexConfig.from_runnable_config(config)

    provider_factory = index_config.provider_factory

    embedding_provider, model_name = extract_provider_and_model(
        index_config.embedding_model
    )
    vstore_provider = index_config.vstore

    logger.info(
        "Save start",
        extra={
            **ctx,
            "segments": len(state.texts + state.imgs + state.tables),
            "vstore_provider": vstore_provider,
            "embedding_provider": embedding_provider,
            "embedding_model": model_name,
        },
    )
    embedding_model = provider_factory.build_embeddings(
        provider=embedding_provider, model_name=model_name
    )
    t_vstore = time.monotonic()
    vstore = await asyncio.to_thread(
        provider_factory.build_vstore,
        embedding_model,
        provider=vstore_provider,
        collection_name=state.collection_id,
        persist_directory=".chroma"
    )
    logger.debug("VStore built", extra={**ctx, "duration_ms": _ms(t_vstore)})

    segments = state.texts + state.imgs + state.tables

    docs = map_to_docs(segments)

    index_docs = filter_complex_metadata(docs)

    if not index_docs:
        logger.warning("No docs to index (skipping)", extra={**ctx, "duration_ms": _ms(start)})
        return {"index_docs": index_docs}

    try:
        await vstore.aadd_documents(index_docs)
    except Exception:
        logger.exception("Indexing failed", extra={**ctx, "docs": len(index_docs)})
        raise

    logger.info(
        "Save done",
        extra={**ctx, "duration_ms": _ms(start), "indexed_docs": len(index_docs)},
    )

    return {"index_docs": index_docs}


def route_by_mode(state: OverallIndexState, config: RunnableConfig) -> list[str]:
    node = "route_by_mode"
    ctx = _log_ctx(state, config, node)

    index_config = IndexConfig.from_runnable_config(config)
    mode = index_config.mode

    if mode == "none":
        nxt = ["save"]
    elif mode == "imgs":
        nxt = ["enrich_imgs_with_llm"]
    elif mode == "tables":
        nxt = ["enrich_tables_with_llm"]
    elif mode == "texts":
        nxt = ["enrich_texts_with_llm"]
    elif mode == "all":
        nxt = [
            "enrich_texts_with_llm",
            "enrich_imgs_with_llm",
            "enrich_tables_with_llm",
        ]
    else:
        logger.error("Unsupported index mode", extra={**ctx, "mode": mode})
        raise ValueError(f"Unsupported index mode: {index_config.mode}")

    logger.info("Routing decided", extra={**ctx, "mode": mode, "next_nodes": nxt})
    return nxt


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
