from collections.abc import Sequence

from langchain_core.documents import Document

from rag_app.index.ocr.schema import Segment


def map_to_docs(data: Sequence[Segment]) -> list[Document]:

    docs = []
    for segment in data:

        if segment.llm_exception is not None: # for llm exceptions no embedding
            continue
        
        llm_meta = segment.llm_metadata
        
        if llm_meta:
            emb_content = llm_meta.retrieval_summary
        else: 
            emb_content = segment.text
        
        metadata = {
            **segment.metadata,
            "llm_language": llm_meta.language if llm_meta else "",
            "llm_title": llm_meta.title if llm_meta else "",
            "llm_labels": llm_meta.labels if llm_meta else "",
            "id": segment.id,
            "source_id": segment.source_id,
            "category": segment.category,
            "page_number": segment.page_number,
            "file_directory": segment.file_directory,
            "filename": segment.filename,
            "text": segment.text,
            "text_as_html": segment.text_as_html,
            "img_url": segment.img_url,
        }

        docs.append(
            Document(
                page_content=emb_content,
                metadata=metadata,
            )
        )

    return docs
