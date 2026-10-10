"""評測結果查詢：排行榜、單一結果、K 值曲線、逐題錯誤分析。"""

from __future__ import annotations

import math
import threading
from collections import OrderedDict

import numpy as np

from embedbench import prefixes
from embedbench.datasets import get_dataset
from embedbench.db import get_db
from embedbench.ranking import Ranking, compute_metrics, ranking_path

_rank_cache: OrderedDict[str, Ranking] = OrderedDict()
_rank_lock = threading.Lock()
RANK_CACHE_SIZE = 16

KIND_LABELS = {
    "embedding": "Embedding",
    "bm25": "BM25",
    "rerank": "Reranker",
    "legacy": "舊資料匯入",
}


def load_ranking(result_id: str) -> Ranking:
    with _rank_lock:
        if result_id in _rank_cache:
            _rank_cache.move_to_end(result_id)
            return _rank_cache[result_id]
    r = Ranking.load(ranking_path(result_id))
    with _rank_lock:
        _rank_cache[result_id] = r
        while len(_rank_cache) > RANK_CACHE_SIZE:
            _rank_cache.popitem(last=False)
    return r


def parse_ks(value) -> list[int]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        value = value.replace("，", ",").split(",")
    return sorted({int(v) for v in value if str(v).strip() and int(v) > 0})


def metrics_for(result: dict, ks: list[int]) -> dict:
    stored = result.get("metrics") or {}
    if all(f"hit@{k}" in stored for k in ks):
        return stored
    return {**stored, **compute_metrics(load_ranking(result["id"]), ks)}


def clean(value):
    """JSON 不支援 NaN / Infinity。"""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


def summarize(result: dict, ks: list[int]) -> dict:
    m = metrics_for(result, ks)
    cfg = result.get("config") or {}
    health = result.get("health") or {}
    truncated = {side: (health.get(side) or {}).get("truncated") or 0 for side in ("query", "document")}
    return clean(
        {
            "id": result["id"],
            "job_id": result.get("job_id"),
            "kind": result["kind"],
            "kind_label": KIND_LABELS.get(result["kind"], result["kind"]),
            "dataset_key": result["dataset_key"],
            "label": result["label"],
            "model": result.get("model"),
            "provider": result.get("provider"),
            "pair_label": result.get("pair_label"),
            "query_prefix": result.get("query_prefix"),
            "document_prefix": result.get("document_prefix"),
            "dims": result.get("dims"),
            "truncated": cfg.get("truncated", False),  # 是否截取向量維度
            "query_limit": cfg.get("query_limit"),
            "truncated_texts": truncated,  # 超過輸入長度上限、自動截短的文字筆數
            "parent_result_id": result.get("parent_result_id"),
            "num_queries": result["num_queries"],
            "num_documents": result["num_documents"],
            "created_at": result["created_at"],
            "metrics": {k: v for k, v in m.items() if any(k == f"{n}@{kk}" for n in ("hit", "mrr", "ndcg", "recall") for kk in ks)},
        }
    )


def list_results(dataset_key: str | None = None, kind: str | None = None, ks: list[int] | None = None) -> list[dict]:
    ks = ks or [1, 3, 5, 10]
    sql, params = "SELECT * FROM results WHERE 1=1", []
    if dataset_key:
        sql += " AND dataset_key = ?"
        params.append(dataset_key)
    if kind:
        sql += " AND kind = ?"
        params.append(kind)
    sql += " ORDER BY created_at DESC"
    return [summarize(r, ks) for r in get_db().query(sql, params, table="results")]


def get_result(result_id: str) -> dict | None:
    return get_db().query_one("SELECT * FROM results WHERE id = ?", (result_id,), table="results")


def result_detail(result_id: str, ks: list[int]) -> dict | None:
    r = get_result(result_id)
    if not r:
        return None
    out = summarize(r, ks)
    out.update(
        clean(
            {
                "config": r.get("config"),
                "health": r.get("health"),
                "timing": r.get("timing"),
                "all_metrics": metrics_for(r, ks),
                "query_prefix_display": prefixes.display(r.get("query_prefix")),
                "document_prefix_display": prefixes.display(r.get("document_prefix")),
            }
        )
    )
    children = get_db().query(
        "SELECT * FROM results WHERE parent_result_id = ? ORDER BY created_at DESC", (result_id,), table="results"
    )
    out["children"] = [summarize(c, ks) for c in children]
    parent = get_result(r["parent_result_id"]) if r.get("parent_result_id") else None
    out["parent"] = summarize(parent, ks) if parent else None
    return out


def k_curve(result_id: str, max_k: int = 20) -> dict:
    r = load_ranking(result_id)
    ks = list(range(1, max_k + 1))
    m = compute_metrics(r, ks)
    return clean({name: [m[f"{name}@{k}"] for k in ks] for name in ("hit", "mrr", "ndcg", "recall")} | {"ks": ks})


def delete_result(result_id: str) -> None:
    get_db().execute("DELETE FROM results WHERE id = ?", (result_id,))
    ranking_path(result_id).unlink(missing_ok=True)
    with _rank_lock:
        _rank_cache.pop(result_id, None)


def query_details(
    result_id: str,
    k: int = 5,
    status: str = "all",  # all | miss | hit
    search: str = "",
    page: int = 1,
    per_page: int = 20,
    top_n: int = 5,
    sort: str = "index",  # index | rank_desc | rank_asc
) -> dict:
    result = get_result(result_id)
    if not result:
        raise KeyError(result_id)
    ds = get_dataset(result["dataset_key"], prepare=False)
    r = load_ranking(result_id)
    qidx = r.queries()
    eval_queries = ds.eval_queries
    best = r.best_ranks()
    order = np.arange(r.num_queries)
    mask = np.ones(r.num_queries, dtype=bool)
    if status == "miss":
        mask &= best > k
    elif status == "hit":
        mask &= best <= k
    if search:
        s = search.strip()
        mask &= np.array([s in eval_queries[int(qidx[i])].text for i in range(r.num_queries)], dtype=bool)
    order = order[mask]
    if sort == "rank_desc":
        order = order[np.argsort(-best[order], kind="stable")]
    elif sort == "rank_asc":
        order = order[np.argsort(best[order], kind="stable")]
    total = len(order)
    page = max(1, page)
    sel = order[(page - 1) * per_page : page * per_page]

    def doc_info(i: int, score=None, rank=None) -> dict:
        d = ds.documents[i]
        return clean(
            {"index": int(i), "id": d.id, "title": d.title, "text": d.text, "score": None if score is None else float(score), "rank": rank}
        )

    items = []
    for i in sel.tolist():
        q = eval_queries[int(qidx[i])]
        gold = r.query_relevant(i)
        gold_set = {g[0] for g in gold}
        top = [
            {**doc_info(int(j), r.top_scores[i, p], p + 1), "relevant": int(j) in gold_set}
            for p, j in enumerate(r.top_idx[i, :top_n])
        ]
        items.append(
            {
                "position": int(i),
                "query_id": q.id,
                "query": q.text,
                "best_rank": int(best[i]),
                "hit": bool(best[i] <= k),
                "gold": [{**doc_info(d), "rank": int(rank), "gain": gain} for d, rank, gain in gold],
                "top": top,
            }
        )
    misses = int((best > k).sum())
    return {
        "total": total,
        "page": page,
        "per_page": per_page,
        "k": k,
        "num_queries": r.num_queries,
        "misses": misses,
        "items": items,
    }
