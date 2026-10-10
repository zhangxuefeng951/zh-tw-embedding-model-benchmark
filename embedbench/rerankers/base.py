"""Reranker 介面：給一個查詢與多個候選文件，回傳每個文件的分數（越高越相關）。"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from embedbench.params import Param, coerce_params, output_signature


class Reranker:
    type_id: str = ""
    label: str = ""
    group: str = "api"  # api | local | llm
    description: str = ""
    params: list[Param] = []

    def __init__(self, params: dict | None = None):
        self.p = coerce_params(self.params, params)
        if not self.p.get("model"):
            raise ValueError("請填寫模型名稱")
        self.check_cancel: Callable[[], None] | None = None

    @classmethod
    def describe(cls) -> dict:
        return {
            "type": cls.type_id,
            "label": cls.label,
            "group": cls.group,
            "description": cls.description,
            "params": [p.to_dict() for p in cls.params],
        }

    @property
    def concurrency(self) -> int:
        """可以同時處理幾個查詢（只有 API 類 reranker 會大於 1）。"""
        return max(1, int(self.p.get("concurrency") or 1)) if self.group in ("api", "llm") else 1

    def score(self, query: str, docs: list[str]) -> np.ndarray:
        raise NotImplementedError

    def close(self) -> None:
        pass


def reranker_signature(cls: type[Reranker], params: dict | None) -> dict:
    values = coerce_params(cls.params, params)
    return {"reranker": cls.type_id, **output_signature(cls.params, values)}
