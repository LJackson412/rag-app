from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, Field

from rag_app.index.schema import LLMMetaData

# NOTE: LLM-Segements are used as Prompt for the LLM.
# They are transferred to the LLM via structured output.


class LLMTextSegment(LLMMetaData):
    extracted_content: Annotated[
        str,
        Field(
            description=(
                "Complete, verbatim, and unmodified text content of this section. "
                "No summarization, no interpretation, and no omitted sentences. "
                "Preserve the original wording, punctuation, and line breaks where they matter "
            ),
        ),
    ]
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


class LLMImageSegment(LLMMetaData):
    extracted_content: Annotated[
        str,
        Field(
            description=(
                "Describe the content of the figure and extract any visible text. "
                "Answer format:\n\n"
                "Figure description:\n"
                "<Put the description of the figure here>\n\n"
                "Extracted text:\n"
                "<Put the original text from the figure here (e.g. axis labels, legends, labels)>"
                "\n\nImportant: Never use this element for tables and lists. Even if a table appears as an image,"
                " extract it as structured HTML and return it in the `tables` list instead."
            ),
        ),
    ]
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


class LLMTableOrListSegment(LLMMetaData):
    extracted_content: Annotated[
        str,
        Field(
            description=(
                "The extracted structure as a complete HTML fragment.\n\n"
                "- For tables: use <table>, <thead>, <tbody>, <tr>, <th>, <td>.\n"
                "- For lists: use <ul> or <ol> with <li> (nested if required).\n\n"
                "The HTML structure should reflect the original layout as closely as possible "
                "(rows, columns, header rows, list hierarchy). "
                "Cell values and list items must be taken verbatim from the document. "
                "Include every tabular structure here, even if the page labels it as a figure or "
                "the table appears as an embedded image."
            ),
        ),
    ]
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


class LLMCodeOrFormulaSegment(LLMMetaData):
    extracted_content: Annotated[
        str,
        Field(
            description=(
                "Verbatim, unmodified content of the code or formula block.\n\n"
                "- For source code: return it as a Markdown code block with the appropriate language, e.g.:\n"
                "  ```python\n"
                "  <original code>\n"
                "  ```\n"
                "- For mathematical formulas: preferably use LaTeX, e.g.:\n"
                "  $$<LaTeX formula>$$\n"
                "- For other notations (e.g. chemical formulas): keep the original notation "
                "without any changes."
            ),
        ),
    ]
    category: Annotated[
        str,
        Field(
            description=(
                "Short category of the block, e.g. "
                "'Code example', 'Configuration snippet', 'Algorithm', "
                "'Mathematical formula', 'Statistical formula', 'Chemical formula'."
            ),
        ),
    ]


class LLMOtherSegment(LLMMetaData):
    extracted_content: Annotated[
        str,
        Field(
            ...,
            description=(
                "Original content of this element as text. "
                "Use this for elements that clearly belong together but do not fit any other category."
            ),
        ),
    ]
    category: Annotated[
        str,
        Field(
            ...,
            description=("Short category for this element, e.g. "),
        ),
    ]


class LLMSegments(BaseModel):
    texts: Annotated[
        list[LLMTextSegment],
        Field(
            default_factory=list,
            description=("List of semantically coherent document text sections"),
        ),
    ]
    figures: Annotated[
        list[LLMImageSegment],
        Field(
            default_factory=list,
            description=("List of detected figures and graphical elements"),
        ),
    ]
    tables: Annotated[
        list[LLMTableOrListSegment],
        Field(
            default_factory=list,
            description=("List of detected tables and lists"),
        ),
    ]
    code_or_formulas: Annotated[
        list[LLMCodeOrFormulaSegment],
        Field(
            default_factory=list,
            description=(
                "List of all detected code blocks and formulas "
                "(programming code, pseudocode, mathematical and other formulas)."
            ),
        ),
    ]
    others: Annotated[
        list[LLMOtherSegment],
        Field(
            default_factory=list,
            description=(
                "List of all content that cannot be clearly assigned to text, table/list, "
                "figure, or code/formula"
            ),
        ),
    ]

# -------------------------------------------------------------------------------

class BaseSegmentAttributes(BaseModel):
    extracted_content: Annotated[
        str,
        Field(
            description=(
                "Note: The extracted content is generated by the OCR model in the OCR graph and by the LLM in the LLM graph."
                "The LLM can hallucinate content."
            ),
        ),
    ]
    metadata: Annotated[
        dict[str, Any],
        Field(
            default_factory=dict,
            description="",
        ),
    ]


# -------------------------------------------------------------------------------


class TextSegment(BaseSegmentAttributes):
    llm_text_segment: Annotated[
        LLMTextSegment,
        Field(
            description=(
                "Structured LLM output for this text section, including extracted "
                "content and the assigned text category."
            ),
        ),
    ]


class ImageSegment(BaseSegmentAttributes):
    llm_image_segment: Annotated[
        LLMImageSegment,
        Field(
            description=(
                "Structured LLM output for this figure, combining the description "
                "and the visual category."
            ),
        ),
    ]


class TableOrListSegment(BaseSegmentAttributes):
    llm_table_segment: Annotated[
        LLMTableOrListSegment,
        Field(
            description=(
                "Structured LLM output for tables or lists, containing the HTML "
                "representation and structural category."
            ),
        ),
    ]


class CodeOrFormulaSegment(BaseSegmentAttributes):
    llm_code_or_formula_segment: Annotated[
        LLMCodeOrFormulaSegment,
        Field(
            description=(
                "Structured LLM output for code or formula blocks, including the "
                "verbatim content and block category."
            ),
        ),
    ]


class OtherSegment(BaseSegmentAttributes):
    llm_other_segment: Annotated[
        LLMOtherSegment,
        Field(
            description=(
                "Structured LLM output for content that does not fit other "
                "segments, capturing the original text and a custom category."
            ),
        ),
    ]
