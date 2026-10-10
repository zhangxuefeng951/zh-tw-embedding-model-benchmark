"""Reranker registry。"""

from __future__ import annotations

from embedbench.params import coerce_params
from embedbench.rerankers.api import LlmListwiseReranker, RerankApi
from embedbench.rerankers.base import Reranker, reranker_signature
from embedbench.rerankers.local import CrossEncoderReranker, FlagReranker, Qwen3Reranker

RERANKERS: dict[str, type[Reranker]] = {
    cls.type_id: cls
    for cls in (RerankApi, LlmListwiseReranker, CrossEncoderReranker, FlagReranker, Qwen3Reranker)
}


def reranker_class(type_id: str) -> type[Reranker]:
    try:
        return RERANKERS[type_id]
    except KeyError:
        raise ValueError(f"未知的 reranker：{type_id}（可用：{', '.join(RERANKERS)}）") from None


def create_reranker(type_id: str, params: dict | None) -> Reranker:
    return reranker_class(type_id)(params)


def signature(type_id: str, params: dict | None) -> dict:
    return reranker_signature(reranker_class(type_id), params)


def normalized_params(type_id: str, params: dict | None) -> dict:
    return coerce_params(reranker_class(type_id).params, params)


def describe_rerankers() -> list[dict]:
    return [cls.describe() for cls in RERANKERS.values()]
