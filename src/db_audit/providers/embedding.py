"""
GenAIHub embedding models
"""

from __future__ import annotations

from typing import Any, Dict, List

import requests  # type: ignore
from langchain_core.embeddings import Embeddings
from pydantic import SecretStr

from db_audit.config.settings import settings


class GenAIHubEmbeddingModel(Embeddings):
    
    def __init__(
        self,
        api_key: SecretStr,
        cert_path: str,
        api_version: str = "2025-01-01-preview",
        model: str = "text-embedding-3-large"
    ):
        self.api_key = api_key
        self.api_version = api_version
        self.cert_path = cert_path
        self.model = model
        self.session = requests.Session()
        
     

    @property
    def url(self) -> str:
        return (
            f"https://genaihub-gateway.genai-prod.comp.db.de/"
            f"openai/deployments/{self.model}/embeddings?"
            f"api-version={self.api_version}"
        )
        
    @property
    def header(self) -> Dict[str, str | None]:
        return {
            "Content-Type": "application/json",
            "Cache-Control": "no-cache",
            "api-key": self.api_key.get_secret_value(),
        }
        

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        
        self.session.verify = self.cert_path
        
        embeddings: List[List[float]]  = []
  
        
        payload: dict[str, list[str] | str] = {
            "input": texts,
            "input_type": "query",
        }
        
        resp = self.session.post(url=self.url, headers=self.header, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        
        items = data.get("data", [])
        items = sorted(items, key=lambda x: x.get("index", 0)) # sortieren um Reihenfolge text/embedding wieder herzustellen
        
        embeddings = [item["embedding"] for item in items]
           
        return embeddings
    

    def embed_query(self, text: str) -> Any:
        """Embed eine Query (z. B. zur Suche)."""
        self.session.verify = self.cert_path
        payload = {"input": text, "input_type": "query"}
        resp = self.session.post(url=self.url, headers=self.header, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["data"][0]["embedding"]

if __name__ == "__main__":
    

    # Test
    embedding_model = GenAIHubEmbeddingModel(
        api_key=settings.GENAIHUB_API_KEY,
        api_version=settings.GENAIHUB_API_VERSION,
        cert_path=settings.GENAIHUB_CERT_PATH
    )
    
    texts = [
        "This is a test document.",
        "This is another document."
    ]
    
    res = embedding_model.embed_documents(texts)
    print(res)
    # -------------------------------------------------    
    