from collections.abc import Sequence
from typing import Any

from langchain_core.documents import Document

from rag_app.index.llm.schema import LLMSegments
from rag_app.index.ocr.schema import (
    BaseSegmentAttributes,
    ImageSegment,
    TableSegment,
    TextSegment,
)


def map_to_docs(data: Sequence[BaseSegmentAttributes]) -> list[Document]:
    docs: list[Document] = []

    def add_chunk(segment: BaseSegmentAttributes, chunk: Any) -> None:
        metadata: dict[str, Any] = {
            **segment.metadata,
            "extracted_content": segment.extracted_content,
            "language": chunk.language,
            "title": chunk.title,
            "labels": chunk.labels,
            "category": chunk.category,
        }

        docs.append(
            Document(
                page_content=chunk.retrieval_summary,
                metadata=metadata,
            )
        )

    for segment in data:
        if isinstance(segment, TextSegment):
            add_chunk(segment, segment.llm_text_segment)
        elif isinstance(segment, ImageSegment):
            add_chunk(segment, segment.llm_image_segment)
        elif isinstance(segment, TableSegment):
            add_chunk(segment, segment.llm_table_segment)

    return docs
