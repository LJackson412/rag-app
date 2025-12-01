EXTRACT_DATA_FROM_PDF_PROMPT = """
You receive a single document page as an image.
Extract all visible content according to the provided schema.\n\n
General rules:\n
- Read all visible text exactly as written (verbatim).
- Do NOT invent, guess, or complete missing content.
- If text is partially unreadable, keep the readable parts and mark the rest as "[UNREADABLE]".
Chunking and structure:\n
- Group content into semantically coherent sections that make sense as standalone RAG chunks.
- Combine heading + directly related paragraph(s) + caption into one `texts` entry if they logically belong together.
- For multi-part elements (e.g. a figure with caption and legend), prefer a single `figure` entry.\n\n
Output requirements:\n
- If a category does not occur on the page, leave the answer blank.
- Always return tables (including table-like screenshots or diagrams with rows and columns) as structured HTML in the `tables` list, not as figures.
"""


GEN_METADATA_PROMPT = """
You receive the plain text content of a single document section:\n\n
Output requirements:\n
- Follow the provided schema exactly.\n
Document section:\n\n
{content}\n
"""
