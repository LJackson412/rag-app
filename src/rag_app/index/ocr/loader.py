from pathlib import Path

from unstructured.chunking.title import chunk_by_title
from unstructured.partition.csv import partition_csv
from unstructured.partition.pdf import partition_pdf

from rag_app.index.ocr.schema import Segment


def load_pdf(path: str, lang: list[str] | None = None) -> list[Segment]:
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

    segments = []
    for e in elements:

        if e.category in ("Image"):
            md = e.metadata.to_dict()

            page_number = md.pop("page_number", -1)
            file_directory = md.pop("file_directory", "")
            filename = md.pop("filename", "")
            img_base64 = md.pop("image_base64", "") or ""
            img_mime_type = md.pop("image_mime_type", "") or ""

            segment = Segment(
                id=None,  # updated later
                source_id=e.id,
                category="Image",
                page_number=page_number,
                file_directory=file_directory,
                filename=filename,
                text=e.text,
                img_base64=img_base64,
                img_mime_type=img_mime_type,
                metadata=md,
            )
            segments.append(segment)

        if e.category in ("Table"):
            md = e.metadata.to_dict()

            page_number = md.pop("page_number", -1)
            file_directory = md.pop("file_directory", "")
            filename = md.pop("filename", "")
            text_as_html = md.pop("text_as_html", "") or ""

            segment = Segment(
                id=None,  # updated later
                source_id=e.id,
                category="Table",
                page_number=page_number,
                file_directory=file_directory,
                filename=filename,
                text=e.text,
                text_as_html=text_as_html,
                metadata=md,
            )
            segments.append(segment)

    filtered_elements = [e for e in elements if e.category not in ("Table", "Image")]
    chunks = chunk_by_title(filtered_elements, include_orig_elements=True)

    for chunk in chunks:
        md = chunk.metadata.to_dict()

        page_number = md.pop("page_number", -1)
        file_directory = md.pop("file_directory", "")
        filename = md.pop("filename", "")

        segment = Segment(
            id=None,  # updated later
            source_id=chunk.id,
            category="Text",
            page_number=page_number,
            file_directory=file_directory,
            filename=filename,
            text=chunk.text,
            metadata=md,
        )
        segments.append(segment)

    return segments


def load_csv(path: str) -> list[Segment]:

    elements = partition_csv(filename=path)

    segments = []
    for e in elements:

        md = e.metadata.to_dict()

        page_number = md.pop("page_number", -1)
        file_directory = md.pop("file_directory", "")
        filename = md.pop("filename", "")
        text_as_html = md.pop("text_as_html", "") or ""

        segment = Segment(
            id=None,  # updated later
            source_id=e.id,
            category="Table",
            page_number=page_number,
            file_directory=file_directory,
            filename=filename,
            text=e.text,
            text_as_html=text_as_html,
            metadata=md,
        )
        segments.append(segment)

    return segments


def load(path: str, lang: list[str] | None = None) -> list[Segment]:
    """
    Load OCR-able files into ``Segment`` objects.

    Supported formats are PDF (``.pdf``) and CSV (``.csv``). The function detects the
    extension case-insensitively and delegates to ``load_pdf`` or ``load_csv``. When the
    extension is not recognized, a ``ValueError`` is raised listing the supported types.
    """

    suffix = Path(path).suffix.lower()

    if suffix == ".pdf":
        return load_pdf(path, lang=lang)
    if suffix == ".csv":
        return load_csv(path)

    supported = [".pdf", ".csv"]
    raise ValueError(f"Unsupported file type: {suffix}. Supported types are {supported}.")
