from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore


def get_chroma_vstore(
    embedding_model: Embeddings,
    collection_name: str = "chroma_collection",
    persist_directory: str = ".chroma_directory",
) -> Chroma:
    return Chroma(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_function=embedding_model,
    )

def get_vstore(
    embedding_model: Embeddings,
    provider: str = "chroma", 
    collection_name: str = "chroma_collection", 
    persist_directory: str = ".chroma_directory"
    ) -> VectorStore:
    match provider:
        case "chroma":
            return get_chroma_vstore(
                embedding_model=embedding_model,
                collection_name=collection_name,
                persist_directory=persist_directory
            )
        case _:
            raise ValueError(f"Unknown provider: {provider}")
