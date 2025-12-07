from __future__ import annotations

from typing import Annotated

from pydantic import Field

from rag_app.index.schema import BaseLLMSegmentAttributes, BaseSegmentAttributes


class LLMTextSegment(BaseLLMSegmentAttributes):
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


class LLMImageSegment(BaseLLMSegmentAttributes):
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


class LLMTableSegment(BaseLLMSegmentAttributes):
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

# -------------------------------------------------------------------------------

class TextSegment(BaseSegmentAttributes):
    llm_text_segment: Annotated[
        LLMTextSegment,
        Field(
            description=(
                "LLM-provided categorization and metadata for this text section "
                "based on the extracted OCR content."
            ),
        ),
    ]


class ImageSegment(BaseSegmentAttributes):
    llm_image_segment: Annotated[
        LLMImageSegment,
        Field(
            description=("List of detected figures and graphical elements"),
        ),
    ]


class TableSegment(BaseSegmentAttributes):
    llm_table_segment: Annotated[
        LLMTableSegment,
        Field(
            description=(
                "List of detected tables and lists with a structured HTML representation."
            ),
        ),
    ]
