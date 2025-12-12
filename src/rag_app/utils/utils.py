import json
from collections.abc import Iterable, Sequence

from langchain_core.documents import Document

DocOrScoredDoc = Document | tuple[Document, float]


def pretty_print_docs(docs: Sequence[DocOrScoredDoc]) -> None:
    """Prints list of document objects (with or without similarity score) with metadata and content formatted."""
    parts: list[str] = []

    for i, item in enumerate(docs):
        d: Document
        score_val: float | None

        # Normalisieren
        if isinstance(item, tuple):
            d, score = item
            score_val = float(score)
        else:
            d = item
            score_val = None

        # Score-Text bauen (wenn vorhanden)
        similarity_line = ""
        if score_val is not None:
            similarity_line = f"\nSimilarity score: {score_val:.4f}"

        metadata_json = json.dumps(d.metadata, indent=2, ensure_ascii=False)

        parts.append(
            f"Document {i + 1}:{similarity_line}\n\n"
            f"Metadata:\n\n{metadata_json}\n\n"
            f"Content:\n\n{d.page_content}"
        )

    print(f"\n{'-' * 100}\n".join(parts))

def return_docs_on_page(
    docs: list[DocOrScoredDoc], page: int, chunk_type: str | None = None
) -> list[DocOrScoredDoc]:
    def get_doc(item: DocOrScoredDoc) -> Document:
        # Wenn es ein (Document, score)-Tuple ist, nimm das Document
        return item[0] if isinstance(item, tuple) else item

    result: list[DocOrScoredDoc] = []
    for item in docs:
        d = get_doc(item)
        if d.metadata.get("page_number") != page:
            continue
        if chunk_type is not None and d.metadata.get("chunk_type") != chunk_type:
            continue
        result.append(item)  # Original beibehalten (inkl. Score, falls vorhanden)
    return result


def attach_metadata(
    docs: Iterable[Document], metadata: dict[str, str]
) -> list[Document]:
    updated_docs: list[Document] = []
    for doc in docs:
        doc.metadata = {**doc.metadata, **metadata}
        updated_docs.append(doc)

    return updated_docs


def make_chunk_id(
    chunk_type: str, collection_id: str, doc_id: str, chunk_index: int
) -> str:
    return f"{chunk_type}::{collection_id}::{doc_id}::{chunk_index:06d}"


def parse_chunk_id(chunk_id: str) -> dict[str, str | int]:
    collection_id, doc_id, idx_str = chunk_id.split("::")
    return {
        "collection_id": collection_id,
        "doc_id": doc_id,
        "chunk_index": int(idx_str),
    }
