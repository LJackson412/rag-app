from dataclasses import dataclass
from pathlib import Path
from typing import Any

from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import Element
from unstructured.partition.csv import partition_csv
from unstructured.partition.docx import partition_docx
from unstructured.partition.pdf import partition_pdf

from rag_app.index.ocr.schema import Segment

COMMON_MD_DEFAULTS = {
    "page_number": -1,
    "file_directory": "",
    "filename": "",
}

CATEGORY_MD_FIELDS: dict[str, dict[str, tuple[str, Any]]] = {
    "Image": {
        "img_base64": ("image_base64", ""),
        "img_mime_type": ("image_mime_type", ""),
    },
    "Table": {
        "text_as_html": ("text_as_html", ""),
    },
    "Text": {},
}


def _segment_from_element(element: Element, category: str) -> Segment:
    md = element.metadata.to_dict()

    # common
    page_number = md.pop("page_number", COMMON_MD_DEFAULTS["page_number"])
    file_directory = md.pop("file_directory", COMMON_MD_DEFAULTS["file_directory"])
    filename = md.pop("filename", COMMON_MD_DEFAULTS["filename"])

    # category-specific
    extra_kwargs: dict[str, Any] = {}
    for seg_field, (md_key, default) in CATEGORY_MD_FIELDS.get(category, {}).items():
        val = md.pop(md_key, default)
        extra_kwargs[seg_field] = val or default  # falls None/"" -> default

    return Segment(
        id=None,  # updated later
        source_id=element.id,
        category=category,
        page_number=page_number,
        file_directory=file_directory,
        filename=filename,
        text=element.text,
        metadata=md,
        **extra_kwargs,
    )


@dataclass(frozen=True)
class ChunkingConfig:
    max_characters: int = 4500  # Hard-Limit: kein Chunk wird länger als das
    new_after_n_chars: int = 3500  # Soft-Limit: lieber neuen Chunk starten
    overlap: int = 0  # Overlap (Zeichen) – standardmäßig NUR bei Splits oversized Elemente
    overlap_all: bool = False  # Overlap auch zwischen “normalen” Chunks anwenden
    combine_text_under_n_chars: int = 600  # Kleine “Pseudo-Titel”-Sektionen zusammenführen
    multipage_sections: bool = True  # Chunks über Seitenumbrüche hinweg
    include_orig_elements: bool = True  # Original-Elemente in chunk.metadata.orig_elements behalten


DEFAULT_CHUNKING = ChunkingConfig()


def load_pdf(
    path: str,
    lang: list[str] | None = None,
    chunking: ChunkingConfig = DEFAULT_CHUNKING,
) -> list[Segment]:
    if lang is None:
        lang = ["eng", "deu"]

    elements = partition_pdf(
        filename=path,
        strategy="hi_res",
        infer_table_structure=True,
        pdf_infer_table_structure=True,
        extract_images_in_pdf=True,
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True,
        languages=lang,
    )

    img_elements = [e for e in elements if e.category == "Image"]
    table_elements = [e for e in elements if e.category == "Table"]
    text_elements = [e for e in elements if e.category not in ("Table", "Image")]

    segments: list[Segment] = []
    segments += [_segment_from_element(e, category="Image") for e in img_elements]

    # Table elements are not merged, regardless of chunk size
    # and if a table is too large, only that table is split internally.
    for e in table_elements:
        for c in chunk_by_title(
            [e],
            max_characters=chunking.max_characters,
            new_after_n_chars=chunking.new_after_n_chars,
            overlap=chunking.overlap,
            overlap_all=chunking.overlap_all,
            combine_text_under_n_chars=0,
            multipage_sections=chunking.multipage_sections,
            include_orig_elements=chunking.include_orig_elements,
        ):
            segments.append(_segment_from_element(c, category="Table"))

    for c in chunk_by_title(
        text_elements,
        max_characters=chunking.max_characters,
        new_after_n_chars=chunking.new_after_n_chars,
        overlap=chunking.overlap,
        overlap_all=chunking.overlap_all,
        combine_text_under_n_chars=chunking.combine_text_under_n_chars,
        multipage_sections=chunking.multipage_sections,
        include_orig_elements=chunking.include_orig_elements,
    ):
        segments.append(_segment_from_element(c, category="Text"))

    return segments


def load_csv(
    path: str,
    chunking: ChunkingConfig = DEFAULT_CHUNKING,
) -> list[Segment]:
    table_elements = partition_csv(filename=path)

    segments: list[Segment] = []
    # Table elements are not merged, regardless of chunk size
    # and if a table is too large, only that table is split internally.
    for e in table_elements:
        for c in chunk_by_title(
            [e],
            max_characters=chunking.max_characters,
            new_after_n_chars=chunking.new_after_n_chars,
            overlap=chunking.overlap,
            overlap_all=chunking.overlap_all,
            combine_text_under_n_chars=0,
            multipage_sections=chunking.multipage_sections,
            include_orig_elements=chunking.include_orig_elements,
        ):
            segments.append(_segment_from_element(c, category="Table"))

    return segments


def load_docx(
    path: str,
    chunking: ChunkingConfig = DEFAULT_CHUNKING,
) -> list[Segment]:
    elements = partition_docx(
        filename=path,
        infer_table_structure=True,
        include_page_breaks=True,
    )

    # Note: Unstructured currently often does not deliver image elements for DOCX.
    # If any do appear, however, they are mapped correctly.
    img_elements = [e for e in elements if e.category == "Image"]
    table_elements = [e for e in elements if e.category == "Table"]
    text_elements = [e for e in elements if e.category not in ("Table", "Image")]

    segments: list[Segment] = []
    segments += [_segment_from_element(e, category="Image") for e in img_elements]

    # Table elements are not merged, regardless of chunk size
    # and if a table is too large, only that table is split internally.
    for e in table_elements:
        for c in chunk_by_title(
            [e],
            max_characters=chunking.max_characters,
            new_after_n_chars=chunking.new_after_n_chars,
            overlap=chunking.overlap,
            overlap_all=chunking.overlap_all,
            combine_text_under_n_chars=0,
            multipage_sections=chunking.multipage_sections,
            include_orig_elements=chunking.include_orig_elements,
        ):
            segments.append(_segment_from_element(c, category="Table"))

    # Text: by-title chunking as in PDF
    for c in chunk_by_title(
        text_elements,
        max_characters=chunking.max_characters,
        new_after_n_chars=chunking.new_after_n_chars,
        overlap=chunking.overlap,
        overlap_all=chunking.overlap_all,
        combine_text_under_n_chars=chunking.combine_text_under_n_chars,
        multipage_sections=chunking.multipage_sections,
        include_orig_elements=chunking.include_orig_elements,
    ):
        segments.append(_segment_from_element(c, category="Text"))

    return segments


def load(path: str, lang: list[str] | None = None) -> list[Segment]:
    suffix = Path(path).suffix.lower()

    if suffix == ".pdf":
        return load_pdf(path, lang=lang)
    if suffix == ".csv":
        return load_csv(path)
    if suffix == ".docx":
        return load_docx(path)

    supported = [".pdf", ".csv", ".docx"]
    raise ValueError(f"Unsupported file type: {suffix}. Supported types are {supported}.")
