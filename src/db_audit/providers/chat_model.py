from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field, SecretStr
from requests import Session  # type: ignore

from db_audit.config.settings import settings


class GenAIHubChatModel(BaseChatModel):
    api_key: SecretStr = Field(..., description="API key for GenAIHub")
    cert_path: str = Field(..., description="")
    api_version: str = Field(default="2025-01-01-preview", description="Azure OpenAI API version")
    session: Session = Field(default_factory=Session, exclude=True)
    model: str = Field(default="gpt-4o", description="Deployed model name")
    temp: float = Field(default=0.0, description="Sampling temperature")

    @property
    def url(self) -> str:
        return (
            f"https://genaihub-gateway.genai-prod.comp.db.de/"
            f"openai/deployments/{self.model}/chat/completions?"
            f"api-version={self.api_version}"
        )
        
    @property
    def header(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Cache-Control": "no-cache",
            "api-key": self.api_key.get_secret_value(),
        }
        
    @property
    def _llm_type(self) -> str:
        return "azure-openai-genaihub"

    def _to_openai_messages(self, messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        """Konvertiert LangChain BaseMessages zu OpenAI-kompatiblen Messages."""
        result = []
        for m in messages:
            role = m.type
            if role == "human":
                role = "user"
            elif role == "ai":
                role = "assistant"
            elif role == "system":
                role = "system"
            # Tool-/Function-Messages kannst du hier bei Bedarf auch mappen
            result.append({"role": role, "content": m.content})
        return result

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        
        if self.model == "gpt-4o-mini":
            payload = {
                "model": self.model,
                "messages": self._to_openai_messages(messages),
                "temperature": self.temp,
                "stop": stop,
            }  
        else: 
            payload = {
                "model": self.model,
                "messages": self._to_openai_messages(messages),
                "stop": stop,
            }
        
        self.session.verify = self.cert_path

        resp = self.session.post(url=self.url, headers=self.header, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage") or {}

        msg = AIMessage(
            content=text,
            usage_metadata={
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
        )

        return ChatResult(generations=[ChatGeneration(message=msg)])


if __name__ == "__main__":
    
    # Test 
    llm = GenAIHubChatModel(
        api_key=settings.GENAIHUB_API_KEY,
        api_version=settings.GENAIHUB_API_VERSION,
        cert_path=settings.GENAIHUB_CERT_PATH
    )
    
    res = llm.invoke("Hello World")
    print(res)
    # -------------------------------------------------

    