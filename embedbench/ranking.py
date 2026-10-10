"""檢索排名與評測指標。

每個評測結果都存成一份 Ranking：
- top_idx / top_scores：每題前 T 名的文件索引與分數（逐題分析、Reranker 第一階段使用）
- 每個正解文件在「完整排名」中的名次與相關分數

有了正解的完整名次，任何 K 值的 Hit / Recall / MRR / NDCG 都可以精確地重新計算，
所以使用者在介面上改 K 值不需要重跑評測。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from embedbench import config

TOP_T = 100  # 每題保留前 100 名
METRIC_NAMES = ("hit", "mrr", "ndcg", "recall")


@dataclass
class Ranking:
    top_idx: np.ndarray  # (nq, T) int32
    top_scores: np.ndarray  # (nq, T) float32
    rel_offsets: np.ndarray  # (nq + 1,) int64
    rel_doc_idx: np.ndarray  # (n_rel,) int32
    rel_rank: np.ndarray  # (n_rel,) int32，1 起算
    rel_gain: np.ndarray  # (n_rel,) float32
    query_index: np.ndarray | None = None  # 每一列對應資料集 eval_queries 的索引；None＝依序 0..nq-1

    @property
    def num_queries(self) -> int:
        return len(self.rel_offsets) - 1

    def queries(self) -> np.ndarray:
        if self.query_index is not None:
            return self.query_index
        return np.arange(self.num_queries)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            "top_idx": self.top_idx,
            "top_scores": self.top_scores,
            "rel_offsets": self.rel_offsets,
            "rel_doc_idx": self.rel_doc_idx,
            "rel_rank": self.rel_rank,
            "rel_gain": self.rel_gain,
        }
        if self.query_index is not None:
            arrays["query_index"] = self.query_index
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: Path) -> Ranking:
        with np.load(path) as z:
            return cls(**{k: z[k] for k in z.files})

    def query_relevant(self, i: int) -> list[tuple[int, int, float]]:
        """第 i 題的正解：[(文件索引, 名次, 相關分數)]，依名次排序。"""
        s, e = self.rel_offsets[i], self.rel_offsets[i + 1]
        items = list(zip(self.rel_doc_idx[s:e].tolist(), self.rel_rank[s:e].tolist(), self.rel_gain[s:e].tolist()))
        return sorted(items, key=lambda x: x[1])

    def best_ranks(self) -> np.ndarray:
        """每題最好的正解名次。"""
        out = np.full(self.num_queries, np.iinfo(np.int32).max, dtype=np.int64)
        counts = np.diff(self.rel_offsets)
        nonempty = counts > 0
        if self.rel_rank.size:
            mins = np.minimum.reduceat(self.rel_rank, self.rel_offsets[:-1][nonempty])
            out[nonempty] = mins
        return out


def results_root() -> Path:
    path = config.data_dir() / "results"
    path.mkdir(parents=True, exist_ok=True)
    return path


def ranking_path(result_id: str) -> Path:
    return results_root() / f"{result_id}.npz"


def normalize(vectors: np.ndarray, dims: int | None = None) -> np.ndarray:
    v = np.asarray(vectors, dtype=np.float32)
    if dims is not None and dims < v.shape[1]:
        v = v[:, :dims]
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return v / norms


def relevance_lists(relevant: list[dict[int, float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    offsets = [0]
    docs, gains = [], []
    for rel in relevant:
        for d, g in rel.items():
            docs.append(d)
            gains.append(g)
        offsets.append(len(docs))
    return (
        np.asarray(offsets, dtype=np.int64),
        np.asarray(docs, dtype=np.int32),
        np.asarray(gains, dtype=np.float32),
    )


def rank_by_scores(
    score_fn: Callable[[int, int], np.ndarray],
    num_queries: int,
    num_docs: int,
    relevant: list[dict[int, float]],
    top_t: int = TOP_T,
    block: int | None = None,
    progress: Callable[[int, int], None] | None = None,
    check_cancel: Callable[[], None] | None = None,
) -> Ranking:
    """score_fn(start, end) 回傳第 start～end-1 題對所有文件的分數矩陣 (end-start, num_docs)。"""
    top_t = min(top_t, num_docs)
    block = block or max(1, min(1024, int(2e7 // max(num_docs, 1))))
    offsets, rel_docs, rel_gain = relevance_lists(relevant)
    rel_rank = np.zeros(len(rel_docs), dtype=np.int32)
    top_idx = np.zeros((num_queries, top_t), dtype=np.int32)
    top_scores = np.zeros((num_queries, top_t), dtype=np.float32)
    for start in range(0, num_queries, block):
        if check_cancel:
            check_cancel()
        end = min(start + block, num_queries)
        scores = np.asarray(score_fn(start, end), dtype=np.float32)
        if top_t < num_docs:
            part = np.argpartition(-scores, top_t - 1, axis=1)[:, :top_t]
        else:
            part = np.tile(np.arange(num_docs), (end - start, 1))
        part_scores = np.take_along_axis(scores, part, axis=1)
        order = np.argsort(-part_scores, axis=1, kind="stable")
        top_idx[start:end] = np.take_along_axis(part, order, axis=1)
        top_scores[start:end] = np.take_along_axis(part_scores, order, axis=1)
        for qi in range(start, end):
            s, e = offsets[qi], offsets[qi + 1]
            if s == e:
                continue
            row = scores[qi - start]
            gold = row[rel_docs[s:e]]
            # 名次 = 1 + 分數嚴格高於正解的文件數
            rel_rank[s:e] = 1 + (row[None, :] > gold[:, None]).sum(axis=1)
        if progress:
            progress(end, num_queries)
    return Ranking(top_idx, top_scores, offsets, rel_docs, rel_rank, rel_gain)


def rank_dense(
    query_vecs: np.ndarray,
    doc_vecs: np.ndarray,
    relevant: list[dict[int, float]],
    dims: int | None = None,
    **kwargs,
) -> Ranking:
    """Cosine 相似度檢索（先正規化再內積）。dims 指定時截取前 dims 維並重新正規化。"""
    q = normalize(query_vecs, dims)
    d = normalize(doc_vecs, dims)
    return rank_by_scores(lambda s, e: q[s:e] @ d.T, len(q), len(d), relevant, **kwargs)


def compute_metrics(r: Ranking, ks: list[int]) -> dict:
    nq = r.num_queries
    out: dict = {"num_queries": nq}
    if nq == 0:
        return out
    counts = np.diff(r.rel_offsets)
    best = r.best_ranks()
    q_of_rel = np.repeat(np.arange(nq), counts)
    discount = 1.0 / np.log2(r.rel_rank.astype(np.float64) + 1.0)
    for k in sorted(set(int(k) for k in ks if int(k) > 0)):
        hit = best <= k
        out[f"hit@{k}"] = float(hit.mean())
        out[f"mrr@{k}"] = float(np.where(hit, 1.0 / best, 0.0).mean())
        within = r.rel_rank <= k
        found = np.bincount(q_of_rel[within], minlength=nq)
        out[f"recall@{k}"] = float((found / np.maximum(counts, 1)).mean())
        dcg = np.bincount(q_of_rel[within], weights=(r.rel_gain * discount)[within], minlength=nq)
        idcg = np.zeros(nq)
        for qi in range(nq):
            s, e = r.rel_offsets[qi], r.rel_offsets[qi + 1]
            if s == e:
                continue
            gains = np.sort(r.rel_gain[s:e])[::-1][:k].astype(np.float64)
            idcg[qi] = (gains / np.log2(np.arange(2, len(gains) + 2))).sum()
        out[f"ndcg@{k}"] = float(np.where(idcg > 0, dcg / np.where(idcg > 0, idcg, 1), 0.0).mean())
    return out


def rerank_ranking(source: Ranking, n_candidates: int, orders: np.ndarray, scores: np.ndarray) -> Ranking:
    """把第一階段前 n 名依 reranker 排序後，接上原本第 n+1 名之後的結果。

    orders[i] 是第 i 題前 n 名候選的新順序（候選位置的排列），scores[i] 對應排序後的分數。
    正解不在候選內時維持原本名次（必定大於 n）。
    """
    nq = source.num_queries
    n = min(n_candidates, source.top_idx.shape[1])
    cand = source.top_idx[:nq, :n]
    new_top = np.take_along_axis(cand, orders[:, :n].astype(np.int64), axis=1)
    top_idx = np.concatenate([new_top, source.top_idx[:nq, n:]], axis=1)
    top_scores = np.concatenate(
        [scores[:, :n].astype(np.float32), np.full((nq, source.top_idx.shape[1] - n), np.nan, dtype=np.float32)],
        axis=1,
    )
    rel_rank = source.rel_rank.copy()
    for qi in range(nq):
        s, e = source.rel_offsets[qi], source.rel_offsets[qi + 1]
        if s == e:
            continue
        pos = {int(d): p + 1 for p, d in enumerate(new_top[qi])}
        for j in range(s, e):
            p = pos.get(int(source.rel_doc_idx[j]))
            if p is not None:
                rel_rank[j] = p
    return Ranking(
        top_idx, top_scores, source.rel_offsets, source.rel_doc_idx, rel_rank, source.rel_gain, source.query_index
    )
