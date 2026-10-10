"""Embedding provider 介面。

新增一種 provider：繼承 EmbeddingProvider，定義 type_id / label / params，實作 embed_batch()，
再加進 providers/__init__.py 的 PROVIDERS。Web UI 表單會依 params 自動產生。
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from embedbench.params import Param, coerce_params, output_signature


class EmbeddingProvider:
    type_id: str = ""
    label: str = ""
    group: str = "api"  # api | local
    description: str = ""
    params: list[Param] = []

    def __init__(self, params: dict | None = None):
        self.p = coerce_params(self.params, params)
        if not self.p.get("model") and self.requires_model:
            raise ValueError("請填寫模型名稱")
        self.check_cancel: Callable[[], None] | None = None
        self._truncations: list[dict] = []
        self._trunc_lock = threading.Lock()

    requires_model = True

    def record_truncation(self, role: str, original: str, kept: str) -> None:
        with self._trunc_lock:
            self._truncations.append(
                {"role": role, "text": original, "original_chars": len(original), "kept_chars": len(kept)}
            )

    def pop_truncations(self) -> list[dict]:
        """取出並清空目前為止被自動截短的紀錄。"""
        with self._trunc_lock:
            out, self._truncations = self._truncations, []
        return out

    @classmethod
    def describe(cls) -> dict:
        return {
            "type": cls.type_id,
            "label": cls.label,
            "group": cls.group,
            "description": cls.description,
            "params": [p.to_dict() for p in cls.params],
        }

    def signature(self) -> dict:
        """會影響向量結果的設定（用來計算快取鍵）。"""
        return {"provider": self.type_id, **output_signature(self.params, self.p)}

    def role_extra(self, role: str) -> dict:
        key = "query_extra" if role == "query" else "document_extra"
        return dict(self.p.get(key) or {})

    def embed_batch(self, texts: list[str], role: str) -> np.ndarray:
        raise NotImplementedError

    def embed(
        self,
        texts: list[str],
        role: str,
        on_progress: Callable[[int], None] | None = None,
        check_cancel: Callable[[], None] | None = None,
    ) -> np.ndarray:
        """分批（必要時並行）計算向量，回傳 (len(texts), dims) 的 float32 陣列。"""
        self.check_cancel = check_cancel
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        bs = max(1, int(self.p.get("batch_size") or 32))
        batches = [texts[i : i + bs] for i in range(0, len(texts), bs)]
        workers = max(1, int(self.p.get("concurrency") or 1)) if self.group == "api" else 1
        results: list[np.ndarray] = []

        def run(batch: list[str]) -> np.ndarray:
            if check_cancel:
                check_cancel()
            out = np.asarray(self.embed_batch(batch, role), dtype=np.float32)
            if out.ndim != 2 or len(out) != len(batch):
                raise RuntimeError(f"回傳的向量數量不符：送出 {len(batch)} 筆，收到 {out.shape}")
            return out

        if workers == 1 or len(batches) == 1:
            for batch in batches:
                results.append(run(batch))
                if on_progress:
                    on_progress(len(batch))
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(run, b) for b in batches]
                try:
                    for batch, fut in zip(batches, futures):
                        results.append(fut.result())
                        if on_progress:
                            on_progress(len(batch))
                except BaseException:
                    for f in futures:
                        f.cancel()
                    raise
        dims = {r.shape[1] for r in results}
        if len(dims) > 1:
            raise RuntimeError(f"不同批次回傳的向量維度不一致：{sorted(dims)}")
        return np.vstack(results)

    def close(self) -> None:
        pass


def parse_embedding_list(items: list, source: str) -> np.ndarray:
    """解析 [{"embedding": [...], "index": i}, ...]（OpenAI 格式），支援 base64。"""
    if not isinstance(items, list):
        raise RuntimeError(f"{source} 回應格式不符：找不到 data 陣列")
    if items and isinstance(items[0], dict) and "index" in items[0]:
        items = sorted(items, key=lambda x: x["index"])
    vectors = []
    for item in items:
        emb = item.get("embedding") if isinstance(item, dict) else item
        if isinstance(emb, str):
            import base64

            emb = np.frombuffer(base64.b64decode(emb), dtype=np.float32)
        vectors.append(np.asarray(emb, dtype=np.float32))
    return np.vstack(vectors) if vectors else np.zeros((0, 0), dtype=np.float32)
