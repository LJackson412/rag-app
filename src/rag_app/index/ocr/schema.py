from dataclasses import dataclass
from typing import Any

from rag_app.index.schema import LLMException, LLMMetaData


@dataclass(slots=True)
class Segment:
    id: str | None # set in graph
    source_id: str # set while loading
    category: str
    page_number: int
    file_directory: str
    filename: str
    metadata: dict[str, Any]
    
    text: str # extracted text from loader
    text_as_html: str | None = None # none for non-table segments
    img_base64: str | None = None  # none for non-image segments
    img_mime_type: str | None  = None # none for non-image segments

    llm_metadata: LLMMetaData | None = None # set in LLM enrichment step
    llm_exception: LLMException | None = None # set in LLM enrichment step

    @property
    def img_url(self) -> str:
        if not self.img_mime_type or not self.img_base64:
            return ""
        return f"data:{self.img_mime_type};base64,{self.img_base64}"
