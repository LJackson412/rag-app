from typing import Sequence, Type, TypeVar, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.messages import HumanMessage
from pydantic import BaseModel


def _build_llm_text_inputs(
    texts: Sequence[str],
    prompt_template: str,
) -> list[LanguageModelInput]:
    inputs: list[LanguageModelInput] = []
    for chunk in texts:
        prompt = prompt_template.format(content=chunk)
        messages = [HumanMessage(content=prompt)]
        inputs.append(messages)
    return inputs


def _build_llm_img_inputs(
    imgs_urls: Sequence[str],
    prompt: str
) -> list[LanguageModelInput]:
    inputs: list[LanguageModelInput] = []
    for img_url in imgs_urls:
        messages = [
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": img_url,
                        },
                    },
                ]
            )
        ]
        inputs.append(messages)
    return inputs


TModel = TypeVar("TModel", bound=BaseModel)


async def gen_llm_structured_data_from_imgs(
    imgs_urls: Sequence[str],
    llm: BaseChatModel,
    gen_prompt: str,
    gen_data: Type[TModel]
) -> list[TModel]:
    """Extrahiere strukturierte Daten aus Text- oder Bild-Chunks."""

    if not imgs_urls:
        return []

    structured_llm = llm.with_structured_output(gen_data)
    inputs = _build_llm_img_inputs(imgs_urls, gen_prompt)

    res = cast(list[TModel], await structured_llm.abatch(inputs))
    return res



async def gen_llm_structured_data_from_texts(
    texts: Sequence[str],
    llm: BaseChatModel,
    gen_prompt: str,
    gen_data: Type[TModel]
) -> list[TModel]:
    """Extrahiere strukturierte Daten aus Text- oder Bild-Chunks."""
    
    if not texts:
        return []

    structured_llm = llm.with_structured_output(gen_data)
    inputs = _build_llm_text_inputs(texts, gen_prompt)

    res = cast(list[TModel], await structured_llm.abatch(inputs))
    return res