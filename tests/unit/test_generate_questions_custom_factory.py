from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any
import pytest
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.rate_limiters import InMemoryRateLimiter

# Ensure the project sources are importable when running tests without installation.
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

os.environ.setdefault("OPENAI_API_KEY", "test")

from rag_app.providers.composition import ProviderFactory
from rag_app.retrieval.graph import generate_questions
from rag_app.retrieval.schema import LLMQuestions
from rag_app.retrieval.state import OverallRetrievalState


class FakeStructuredModel:
    def __init__(self, output: LLMQuestions):
        self.output: LLMQuestions = output

    def __call__(self, _input: Any, config: RunnableConfig | None = None) -> LLMQuestions:
        return self.output

    def invoke(self, _input: Any, config: RunnableConfig | None = None) -> LLMQuestions:
        return self.output

    async def ainvoke(
        self, _input: Any, config: RunnableConfig | None = None
    ) -> LLMQuestions:
        return self.output


class FakeChatModel:
    def __init__(self, output: LLMQuestions):
        self.output: LLMQuestions = output

    def with_structured_output(self, _schema: type[LLMQuestions]) -> FakeStructuredModel:
        return FakeStructuredModel(self.output)


class FakeProviderFactory(ProviderFactory):
    def __init__(self, output: LLMQuestions):
        self.output: LLMQuestions = output
        self.invoked: bool = False

    def build_chat_model(
        self,
        model_name: str = "gpt-4.1",
        provider: str = "openai",
        temp: float = 0.0,
        max_retries: int = 5,
        rate_limiter: InMemoryRateLimiter | None = None,
    ) -> FakeChatModel:
        self.invoked = True
        return FakeChatModel(self.output)


def test_generate_questions_uses_custom_provider_factory() -> None:
    stubbed_questions = LLMQuestions(questions=["q1", "q2", "q3"])
    fake_factory = FakeProviderFactory(output=stubbed_questions)

    state = OverallRetrievalState(messages=[HumanMessage(content="What is LangGraph?")])
    config = RunnableConfig(configurable={"provider_factory": fake_factory})

    llm_output = asyncio.run(generate_questions(state, config))

    assert llm_output == {"llm_questions": stubbed_questions.questions}
    assert fake_factory.invoked is True
