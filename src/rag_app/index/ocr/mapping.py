from typing import Any

from langchain_core.documents import Document

from rag_app.index.ocr.schema import BaseAttributes, DocumentSegment


def map_to_docs(data: list[DocumentSegment]) -> list[Document]:
    docs: list[Document] = []

    def add_chunk(segment: DocumentSegment, chunk: BaseAttributes) -> None:
        
        base_metadata: dict[str, Any] = {
            **segment.metadata,
            "extracted_content": segment.extracted_content,
        }

        chunk_dict = chunk.model_dump()
        page_content = chunk_dict["retrieval_summary"]

        metadata: dict[str, Any] = {
            **base_metadata,
            **chunk_dict,
        }

        docs.append(
            Document(
                page_content=page_content,
                metadata=metadata,
            )
        )

    for segment in data:
        if segment.text is not None:
            add_chunk(segment, segment.text)
        if segment.figure is not None:
            add_chunk(segment, segment.figure)
        if segment.table is not None:
            add_chunk(segment, segment.table)

    return docs
