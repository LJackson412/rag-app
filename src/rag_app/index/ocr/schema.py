from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field


class BaseAttributes(BaseModel):
    language: Annotated[
        Literal["de", "eng", "n/a"],
        Field(
            description="Detected document segment language; all outputs must use this language.",
        ),
    ]
    title: Annotated[
        str,
        Field(
            description=(
                "Original title of this document section, if present. "
                "If the section has no explicit title, create a short, precise, and informative title "
                "that clearly describes the content of this section."
            ),
        ),
    ]
    retrieval_summary: Annotated[
        str,
        Field(
            description=(
                "Write a short, retrieval-oriented summary of this document section in the same language as the original text."
                "Capture the main topic, important entities and concepts, their relationships, and any crucial numbers, dates, or identifiers. "
                "Use key domain terms and phrases that users might search for. 1-3 sentences, no bullet points."
            ),
        ),
    ]
    labels: Annotated[
        list[str],
        Field(
            default_factory=list,
            description=(
                "List of 2-3 short tags to enrich this section with metadata, "
            ),
        ),
    ]


class Text(BaseAttributes):
    category: Annotated[
        str,
        Field(
            description=(
                "Short textual category describing the text type, e.g. "
                "'Heading', 'Subheading', 'Body text', 'Bullet list', "
                "'Footnote', 'Quote', 'Caption'."
            ),
        ),
    ]


class Figure(BaseAttributes):
    category: Annotated[
        str,
        Field(
            description=(
                "Short label for the visual type of the figure, e.g. "
                "'Bar chart', 'Line chart', 'Pie chart', 'Photo', 'Logo', "
                "'Screenshot', 'Technical drawing'. "
                "Do not use a whole PDF page as a category, and never classify tables here."
            ),
        ),
    ]


class Table(BaseAttributes):
    category: Annotated[
        str,
        Field(
            description=(
                "Short label for the structural type, e.g. "
                "'Table', 'Financial table', 'Matrix', 'Data overview', "
                "'Bullet list', 'Numbered list', 'Checklist'."
            ),
        ),
    ]


class DocumentSegment(BaseModel):
    extracted_content: Annotated[
        str,
        Field(
            description="",
        ),
    ]
    metadata: Annotated[
        dict[str, Any],
        Field(
            default_factory=dict,
            description="Additional metadata such as source, page number or PDF information.",
        ),
    ]
    text: Annotated[
        Text | None,
        Field(
            default=None,
            description=(
                "List of semantically coherent document sections "
                "(e.g. heading plus associated paragraph, individual paragraphs, captions, lists). "
                "Split the content so that each entry is a meaningful RAG chunk "
                "for question-answering systems."
            ),
        ),
    ]
    figure: Annotated[
        Figure | None,
        Field(
            default=None,
            description=("List of detected figures and graphical elements"),
        ),
    ]
    table: Annotated[
        Table | None,
        Field(
            default=None,
            description=(
                "List of detected tables and lists with a structured HTML representation."
            ),
        ),
    ]
