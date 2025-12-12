from collections.abc import Sequence
from typing import Any

from langchain_core.documents import Document

from rag_app.index.llm.schema import (
    BaseSegmentAttributes,
    CodeOrFormulaSegment,
    ImageSegment,
    OtherSegment,
    TableOrListSegment,
    TextSegment,
)


def map_to_docs(data: Sequence[BaseSegmentAttributes]) -> list[Document]:
    docs: list[Document] = []

    def add_chunk(segment: BaseSegmentAttributes, chunk: Any) -> None:
        metadata: dict[str, Any] = {
            **segment.metadata,
            "extracted_content": chunk.extracted_content,
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
        elif isinstance(segment, TableOrListSegment):
            add_chunk(segment, segment.llm_table_segment)
        elif isinstance(segment, CodeOrFormulaSegment):
            add_chunk(segment, segment.llm_code_or_formula_segment)
        elif isinstance(segment, OtherSegment):
            add_chunk(segment, segment.llm_other_segment)

    return docs
