GENERATE_QUESTIONS_PROMPT = """
Your task is to generate {number} additional query variants for semantic search in a vector database, based on the following user question.

Guidelines:
- Always answer in the same language as the original question.

Question:
{question}
"""

COMPRESS_DOCS_PROMPT = """
Evaluate whether the following document segment is relevant to the question.\n

Question:
{question}

Document Segment:
{doc_content}
"""

GENERATE_ANSWER_PROMPT = """
Using the relevant document segments excerpts below, answer the users question following the defined answer schema.

Guidelines:
- Respond in the same language as the question.
- Base your answer only on the provided document segments.

Question:
{question}

Document Segments:\n\n
{docs}
"""
