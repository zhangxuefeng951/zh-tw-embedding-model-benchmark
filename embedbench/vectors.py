"""分段存放、可續跑的陣列快取（向量、reranker 分數都用它）。

每一段存成 <start>-<end>.npy，代表第 start～end-1 筆。中斷後重跑時，從已完成的最後一筆繼續；
快取鍵由「會影響結果的設定」計算而來，因此只有設定完全相同時才會共用。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np

from embedbench import config

_CHUNK = re.compile(r"^(\d{9})-(\d{9})\.npy$")


def cache_key(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


class ChunkStore:
    def __init__(self, directory: Path, meta: dict | None = None):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        meta_path = directory / "meta.json"
        if meta is not None and not meta_path.exists():
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2, default=str)

    def _chunks(self) -> list[tuple[int, int, Path]]:
        out = []
        for p in self.directory.iterdir():
            m = _CHUNK.match(p.name)
            if m:
                out.append((int(m.group(1)), int(m.group(2)), p))
        out.sort()
        return out

    def count(self) -> int:
        """從 0 開始連續完成的筆數。"""
        done = 0
        for start, end, _ in self._chunks():
            if start > done:
                break
            done = max(done, end)
        return done

    def append(self, start: int, array: np.ndarray, info: dict | None = None) -> None:
        """寫入第 start 筆起的一段；info（例如被截短的索引）存成同名的 .json。"""
        end = start + len(array)
        final = self.directory / f"{start:09d}-{end:09d}.npy"
        if info:
            info_tmp = self.directory / f".tmp-{start:09d}-{os.getpid()}.json"
            with open(info_tmp, "w", encoding="utf-8") as f:
                json.dump(info, f, ensure_ascii=False)
            os.replace(info_tmp, final.with_suffix(".json"))
        tmp = self.directory / f".tmp-{start:09d}-{os.getpid()}.npy"
        np.save(tmp, array)
        os.replace(tmp, final)

    def truncated(self, n: int) -> list[int]:
        """前 n 筆中，計算向量時被自動截短的索引。"""
        out = set()
        for start, _end, path in self._chunks():
            info_path = path.with_suffix(".json")
            if start >= n or not info_path.exists():
                continue
            with open(info_path, encoding="utf-8") as f:
                info = json.load(f)
            out.update(start + i for i in info.get("truncated", []) if start + i < n)
        return sorted(out)

    def load(self, n: int) -> np.ndarray:
        parts, done = [], 0
        for start, end, path in self._chunks():
            if done >= n:
                break
            if start > done:
                break
            if end <= done:
                continue
            arr = np.load(path)
            parts.append(arr[done - start : min(end, n) - start])
            done = min(end, n)
        if done < n:
            raise RuntimeError(f"快取只有 {done} 筆，需要 {n} 筆")
        return np.concatenate(parts) if parts else np.zeros((0,))


def vectors_root() -> Path:
    path = config.data_dir() / "vectors"
    path.mkdir(parents=True, exist_ok=True)
    return path


def store_for(kind: str, payload: dict) -> ChunkStore:
    key = cache_key(payload)
    return ChunkStore(config.data_dir() / kind / key, meta=payload)


def clear_cache(kind: str = "vectors") -> int:
    root = config.data_dir() / kind
    if not root.exists():
        return 0
    n = sum(1 for p in root.iterdir() if p.is_dir())
    shutil.rmtree(root)
    return n


def vector_health(vectors: np.ndarray) -> dict:
    """向量健康檢查：NaN、無限大、全零向量、長度分布。"""
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or len(vectors) == 0:
        return {"count": int(len(vectors)), "dims": None}
    finite = np.isfinite(vectors)
    norms = np.linalg.norm(np.where(finite, vectors, 0), axis=1)
    return {
        "count": int(len(vectors)),
        "dims": int(vectors.shape[1]),
        "nan_rows": int(np.isnan(vectors).any(axis=1).sum()),
        "inf_rows": int(np.isinf(vectors).any(axis=1).sum()),
        "zero_rows": int((norms == 0).sum()),
        "norm_min": float(norms.min()),
        "norm_mean": float(norms.mean()),
        "norm_max": float(norms.max()),
    }
