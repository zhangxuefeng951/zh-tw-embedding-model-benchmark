"""Embedding provider registry。"""

from __future__ import annotations

from embedbench.params import coerce_params, output_signature
from embedbench.providers.api import (
    CohereProvider,
    GeminiProvider,
    OpenAICompatProvider,
    VertexProvider,
    VoyageProvider,
)
from embedbench.providers.base import EmbeddingProvider
from embedbench.providers.local import (
    FlagEmbeddingProvider,
    Model2VecProvider,
    SentenceTransformersProvider,
    TransformersProvider,
)

PROVIDERS: dict[str, type[EmbeddingProvider]] = {
    cls.type_id: cls
    for cls in (
        OpenAICompatProvider,
        CohereProvider,
        VoyageProvider,
        GeminiProvider,
        VertexProvider,
        SentenceTransformersProvider,
        TransformersProvider,
        FlagEmbeddingProvider,
        Model2VecProvider,
    )
}


def provider_class(type_id: str) -> type[EmbeddingProvider]:
    try:
        return PROVIDERS[type_id]
    except KeyError:
        raise ValueError(f"未知的 embedding provider：{type_id}（可用：{', '.join(PROVIDERS)}）") from None


def create_provider(type_id: str, params: dict | None) -> EmbeddingProvider:
    return provider_class(type_id)(params)


def provider_signature(type_id: str, params: dict | None) -> dict:
    """不必載入模型就能算出「會影響向量的設定」，用來判斷快取能不能共用。"""
    cls = provider_class(type_id)
    values = coerce_params(cls.params, params)
    return {"provider": type_id, **output_signature(cls.params, values)}


def normalized_params(type_id: str, params: dict | None) -> dict:
    cls = provider_class(type_id)
    return coerce_params(cls.params, params)


def describe_providers() -> list[dict]:
    return [cls.describe() for cls in PROVIDERS.values()]
