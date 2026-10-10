"""評測執行器：embedding、BM25、reranker、資料集下載、舊資料匯入。

Web UI 的任務佇列與 CLI 都呼叫這裡的函式，兩者行為一致。
"""

from __future__ import annotations

import secrets
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np

from embedbench import prefixes
from embedbench.bm25 import cached_index, tokenize
from embedbench.datasets import dataset_exists, get_dataset, prepare_dataset
from embedbench.db import Database, get_db, now_iso
from embedbench.http import ApiError
from embedbench.providers import create_provider, normalized_params, provider_signature
from embedbench.ranking import (
    TOP_T,
    Ranking,
    compute_metrics,
    rank_by_scores,
    rank_dense,
    ranking_path,
    rerank_ranking,
)
from embedbench.rerankers import create_reranker
from embedbench.rerankers import normalized_params as reranker_params
from embedbench.rerankers import signature as reranker_signature
from embedbench.truncation import is_context_error
from embedbench.vectors import store_for, vector_health

STORED_KS = [1, 3, 5, 10, 20, 50, 100]


class Cancelled(Exception):
    pass


class JobContext:
    def __init__(
        self,
        job_id: str | None = None,
        cancel_event: threading.Event | None = None,
        on_progress: Callable[[dict], None] | None = None,
    ):
        self.job_id = job_id
        self.cancel_event = cancel_event or threading.Event()
        self.on_progress = on_progress
        self.messages: list[str] = []
        self._lock = threading.Lock()

    def check_cancel(self) -> None:
        if self.cancel_event.is_set():
            raise Cancelled()

    def progress(self, stage: str, done: int = 0, total: int = 0) -> None:
        if self.on_progress:
            with self._lock:
                self.on_progress({"stage": stage, "done": int(done), "total": int(total), "messages": self.messages})

    def log(self, message: str) -> None:
        self.messages.append(message)
        self.progress(message)


def new_id(prefix: str) -> str:
    return f"{prefix}{datetime.now():%Y%m%d%H%M%S}{secrets.token_hex(3)}"


def save_result(db: Database, ranking: Ranking, row: dict) -> str:
    rid = new_id("r")
    ranking.save(ranking_path(rid))
    db.insert(
        "results",
        {
            "id": rid,
            "num_queries": ranking.num_queries,
            "metrics": compute_metrics(ranking, STORED_KS),
            "created_at": now_iso(),
            **row,
        },
    )
    return rid


# ---- 設定驗證 ------------------------------------------------------------


def _datasets(cfg: dict) -> list[str]:
    keys = cfg.get("datasets") or ([cfg["dataset"]] if cfg.get("dataset") else [])
    if isinstance(keys, str):
        keys = [keys]
    keys = list(dict.fromkeys(k for k in keys if k))
    if not keys:
        raise ValueError("請至少選擇一個資料集")
    for k in keys:
        if not dataset_exists(k):
            raise ValueError(f"找不到資料集：{k}")
    return keys


def _limit(value) -> int | None:
    if value in (None, "", 0, "0"):
        return None
    n = int(value)
    return n if n > 0 else None


def pair_label(pair: dict) -> str:
    q, d = pair.get("query") or "", pair.get("document") or ""
    if not q and not d:
        return "無前綴"
    return f"Q「{prefixes.display(q)[:24]}」D「{prefixes.display(d)[:24]}」"


def _fill_from_preset(preset_id: str | None, provider: str, params: dict) -> dict:
    """截斷用的 tokenizer / 上限沒填時，從預設檔補上（不影響向量，只影響過長文字的處理）。"""
    if not preset_id:
        return params
    from embedbench.presets import get_preset

    preset = get_preset("embedding", preset_id)
    if not preset:
        return params
    source = preset.get("params") if preset.get("provider") == provider else (preset.get("provider_params") or {}).get(provider)
    for key in ("tokenizer", "max_input_tokens"):
        if key in params and not params.get(key) and (source or {}).get(key):
            params[key] = source[key]
    return params


def normalize_embedding_config(cfg: dict) -> dict:
    provider = cfg.get("provider") or "openai"
    params = normalized_params(provider, cfg.get("params"))
    if not params.get("model"):
        raise ValueError("請填寫模型名稱")
    params = _fill_from_preset(cfg.get("preset_id"), provider, params)
    pairs, seen = [], set()
    for p in cfg.get("prefix_pairs") or []:
        q, d = p.get("query") or "", p.get("document") or ""
        if (q, d) in seen:
            continue
        seen.add((q, d))
        pairs.append({"label": (p.get("label") or "").strip() or pair_label(p), "query": q, "document": d})
    if not pairs:
        pairs = [{"label": "無前綴", "query": "", "document": ""}]
    dims = cfg.get("eval_dims") or []
    if isinstance(dims, str):
        dims = [x for x in dims.replace("，", ",").split(",") if x.strip()]
    eval_dims = sorted({int(x) for x in dims if int(x) > 0}, reverse=True)
    return {
        "label": (cfg.get("label") or "").strip() or params["model"],
        "preset_id": cfg.get("preset_id"),
        "provider": provider,
        "params": params,
        "prefix_pairs": pairs,
        "datasets": _datasets(cfg),
        "eval_dims": eval_dims,
        "query_limit": _limit(cfg.get("query_limit")),
    }


def normalize_bm25_config(cfg: dict) -> dict:
    return {
        "label": (cfg.get("label") or "").strip() or "BM25（jieba）",
        "datasets": _datasets(cfg),
        "k1": float(cfg.get("k1") or 1.5),
        "b": float(cfg.get("b") if cfg.get("b") not in (None, "") else 0.75),
        "query_limit": _limit(cfg.get("query_limit")),
    }


def normalize_rerank_config(cfg: dict) -> dict:
    db = get_db()
    source_id = cfg.get("source_result_id")
    source = db.query_one("SELECT id, label FROM results WHERE id = ?", (source_id,)) if source_id else None
    if not source:
        raise ValueError("請選擇第一階段的檢索結果")
    reranker = cfg.get("reranker") or "rerank_api"
    params = reranker_params(reranker, cfg.get("params"))
    if not params.get("model"):
        raise ValueError("請填寫 reranker 模型名稱")
    candidates = int(cfg.get("candidates") or 50)
    if not 1 <= candidates <= TOP_T:
        raise ValueError(f"候選數量必須介於 1～{TOP_T}")
    return {
        "label": (cfg.get("label") or "").strip() or f"{params['model']} ← {source['label']}",
        "preset_id": cfg.get("preset_id"),
        "source_result_id": source_id,
        "reranker": reranker,
        "params": params,
        "candidates": candidates,
        "include_title": bool(cfg.get("include_title")),
    }


def normalize_prepare_config(cfg: dict) -> dict:
    key = cfg.get("dataset")
    if not key or not dataset_exists(key):
        raise ValueError(f"找不到資料集：{key}")
    return {"dataset": key, "force": bool(cfg.get("force"))}


def normalize_legacy_config(cfg: dict) -> dict:
    from pathlib import Path

    path = (cfg.get("path") or "").strip()
    if not path or not Path(path).expanduser().exists():
        raise ValueError(f"找不到舊資料庫檔案：{path}")
    return {
        "path": str(Path(path).expanduser().resolve()),
        "dataset": cfg.get("dataset") or "tceval-drcd-test",
        "models": cfg.get("models") or None,
    }


# ---- Embedding 評測 ------------------------------------------------------


def run_embedding(cfg: dict, ctx: JobContext) -> list[str]:
    cfg = normalize_embedding_config(cfg)
    db = get_db()
    sig = provider_signature(cfg["provider"], cfg["params"])
    provider = None
    result_ids: list[str] = []
    try:
        for ds_key in cfg["datasets"]:
            ctx.check_cancel()
            ds = get_dataset(ds_key, progress=lambda st, d, t: ctx.progress(f"準備資料集：{st}", d, t))
            queries = ds.eval_queries[: cfg["query_limit"]] if cfg["query_limit"] else ds.eval_queries
            relevant = [ds.relevant(q.id) for q in queries]

            needed: list[tuple[str, str]] = []
            for pair in cfg["prefix_pairs"]:
                for item in (("document", pair["document"]), ("query", pair["query"])):
                    if item not in needed:
                        needed.append(item)

            vectors: dict[tuple[str, str], np.ndarray] = {}
            timing: dict[tuple[str, str], dict] = {}
            truncated: dict[tuple[str, str], int] = {}
            for role, template in needed:
                if role == "document":
                    texts = [prefixes.render(template, d.text, d.title) for d in ds.documents]
                    content, n = ds.documents_hash, len(texts)
                else:
                    texts = [prefixes.render(template, q.text) for q in queries]
                    content, n = ds.queries_hash, len(texts)
                store = store_for(
                    "vectors", {**sig, "role": role, "template": template, "dataset": ds.key, "content": content}
                )
                cached = store.count()
                t0 = time.time()
                if cached < n:
                    if provider is None:
                        ctx.progress("載入模型 / 建立連線")
                        provider = create_provider(cfg["provider"], cfg["params"])
                    name = "文件" if role == "document" else "查詢"
                    stage = f"[{ds.name}] 計算{name}向量（前綴：{prefixes.display(template)}）"
                    bs = int(cfg["params"].get("batch_size") or 32)
                    # 每段算完就存檔；段落小一點，中斷時損失較少
                    chunk = max(64, bs * int(cfg["params"].get("concurrency") or 1))
                    counter = cached
                    ctx.progress(stage, counter, n)

                    def on_progress(k: int, stage=stage, n=n) -> None:
                        nonlocal counter
                        counter += k
                        ctx.progress(stage, counter, n)

                    for start in range(cached, n, chunk):
                        end = min(start + chunk, n)
                        arr = provider.embed(texts[start:end], role, on_progress=on_progress, check_cancel=ctx.check_cancel)
                        for notice in provider.pop_notices():
                            ctx.log(f"[{ds.name}] {notice}")
                        originals = {t["text"] for t in provider.pop_truncations()}
                        local = [i for i, t in enumerate(texts[start:end]) if t in originals]
                        store.append(start, arr, {"truncated": local} if local else None)
                vectors[(role, template)] = store.load(n)
                timing[(role, template)] = {"seconds": round(time.time() - t0, 2), "cached": min(cached, n), "total": n}
                truncated[(role, template)] = len(store.truncated(n))
                if truncated[(role, template)]:
                    name = "文件" if role == "document" else "查詢"
                    ctx.log(
                        f"[{ds.name}] {name}（前綴：{prefixes.display(template)}）有 {truncated[(role, template)]} 筆"
                        "超過模型長度上限，已自動從尾端截短（保留前綴與開頭）"
                    )

            for pair in cfg["prefix_pairs"]:
                qv = vectors[("query", pair["query"])]
                dv = vectors[("document", pair["document"])]
                if qv.shape[1] != dv.shape[1]:
                    raise RuntimeError(f"查詢向量 {qv.shape[1]} 維與文件向量 {dv.shape[1]} 維不一致")
                full = int(dv.shape[1])
                health = {
                    "query": {**vector_health(qv), "truncated": truncated[("query", pair["query"])]},
                    "document": {**vector_health(dv), "truncated": truncated[("document", pair["document"])]},
                }
                for side, h in health.items():
                    if h.get("nan_rows") or h.get("inf_rows"):
                        ctx.log(f"警告：{pair['label']} 的{side}向量有 NaN / 無限大（{h}）")
                    if h.get("zero_rows"):
                        ctx.log(f"警告：{pair['label']} 的{side}向量有 {h['zero_rows']} 筆全零")
                dims_list: list[int | None] = []
                for d in cfg["eval_dims"] or [None]:
                    if d is not None and d > full:
                        ctx.log(f"略過評測維度 {d}：模型輸出只有 {full} 維")
                        continue
                    use = None if d is None or d == full else d
                    if use not in dims_list:
                        dims_list.append(use)
                for use in dims_list:
                    used = use or full
                    stage = f"[{ds.name}] 檢索評測：{pair['label']}（{used} 維）"
                    t0 = time.time()
                    ranking = rank_dense(
                        qv,
                        dv,
                        relevant,
                        dims=use,
                        progress=lambda d, t, stage=stage: ctx.progress(stage, d, t),
                        check_cancel=ctx.check_cancel,
                    )
                    tq, td = timing[("query", pair["query"])], timing[("document", pair["document"])]
                    rid = save_result(
                        db,
                        ranking,
                        {
                            "job_id": ctx.job_id,
                            "kind": "embedding",
                            "dataset_key": ds.key,
                            "label": cfg["label"],
                            "model": cfg["params"]["model"],
                            "provider": cfg["provider"],
                            "pair_label": pair["label"],
                            "query_prefix": pair["query"],
                            "document_prefix": pair["document"],
                            "dims": used,
                            "num_documents": len(ds.documents),
                            "config": {
                                "provider": cfg["provider"],
                                "params": cfg["params"],
                                "preset_id": cfg["preset_id"],
                                "pair": pair,
                                "full_dims": full,
                                "truncated": use is not None,
                                "query_limit": cfg["query_limit"],
                                "dataset_hash": ds.documents_hash,
                                "queries_hash": ds.queries_hash,
                            },
                            "health": health,
                            "timing": {
                                "query_embed": tq,
                                "document_embed": td,
                                "rank_seconds": round(time.time() - t0, 2),
                            },
                        },
                    )
                    result_ids.append(rid)
    finally:
        if provider is not None:
            provider.close()
    return result_ids


# ---- BM25 ---------------------------------------------------------------


def run_bm25(cfg: dict, ctx: JobContext) -> list[str]:
    cfg = normalize_bm25_config(cfg)
    db = get_db()
    result_ids = []
    for ds_key in cfg["datasets"]:
        ctx.check_cancel()
        ds = get_dataset(ds_key, progress=lambda st, d, t: ctx.progress(f"準備資料集：{st}", d, t))
        queries = ds.eval_queries[: cfg["query_limit"]] if cfg["query_limit"] else ds.eval_queries
        relevant = [ds.relevant(q.id) for q in queries]
        t0 = time.time()
        index = cached_index(
            ds.key,
            ds.documents_hash,
            [d.text for d in ds.documents],
            cfg["k1"],
            cfg["b"],
            progress=lambda d, t: ctx.progress(f"[{ds.name}] jieba 斷詞（文件）", d, t),
        )
        ctx.check_cancel()
        qmat = index.query_matrix(
            tokenize([q.text for q in queries], lambda d, t: ctx.progress(f"[{ds.name}] jieba 斷詞（查詢）", d, t))
        )
        stage = f"[{ds.name}] BM25 檢索評測"
        ranking = rank_by_scores(
            lambda s, e: index.scores(qmat[s:e]),
            len(queries),
            len(ds.documents),
            relevant,
            progress=lambda d, t: ctx.progress(stage, d, t),
            check_cancel=ctx.check_cancel,
        )
        result_ids.append(
            save_result(
                db,
                ranking,
                {
                    "job_id": ctx.job_id,
                    "kind": "bm25",
                    "dataset_key": ds.key,
                    "label": cfg["label"],
                    "model": "BM25Okapi",
                    "provider": "bm25",
                    "pair_label": f"jieba 斷詞，k1={cfg['k1']}，b={cfg['b']}",
                    "query_prefix": "",
                    "document_prefix": "",
                    "dims": None,
                    "num_documents": len(ds.documents),
                    "config": {**cfg, "dataset_hash": ds.documents_hash},
                    "timing": {"seconds": round(time.time() - t0, 2)},
                },
            )
        )
    return result_ids


# ---- Reranker 二階段評測 --------------------------------------------------


def run_rerank(cfg: dict, ctx: JobContext) -> list[str]:
    cfg = normalize_rerank_config(cfg)
    db = get_db()
    source = db.query_one("SELECT * FROM results WHERE id = ?", (cfg["source_result_id"],), table="results")
    ds = get_dataset(source["dataset_key"], progress=lambda st, d, t: ctx.progress(f"準備資料集：{st}", d, t))
    src_ranking = Ranking.load(ranking_path(source["id"]))
    qidx = src_ranking.queries()
    eval_queries = ds.eval_queries
    n = min(cfg["candidates"], src_ranking.top_idx.shape[1])
    nq = src_ranking.num_queries

    def doc_text(i: int) -> str:
        d = ds.documents[i]
        return f"{d.title}\n{d.text}" if cfg["include_title"] and d.title else d.text

    store = store_for(
        "rerank",
        {
            **reranker_signature(cfg["reranker"], cfg["params"]),
            "source": source["id"],
            "n": n,
            "include_title": cfg["include_title"],
            "dataset": ds.key,
            "content": ds.documents_hash,
        },
    )
    cached = store.count()
    t0 = time.time()
    reranker = None
    try:
        if cached < nq:
            ctx.progress("載入 reranker / 建立連線")
            reranker = create_reranker(cfg["reranker"], cfg["params"])
            reranker.check_cancel = ctx.check_cancel
            stage = f"[{ds.name}] Reranker 重新排序前 {n} 名"

            def one(i: int) -> np.ndarray:
                ctx.check_cancel()
                cand = src_ranking.top_idx[i, :n]
                q = eval_queries[int(qidx[i])]
                scores = np.asarray(reranker.score(q.text, [doc_text(int(j)) for j in cand]), dtype=np.float64)
                if scores.shape != (len(cand),):
                    raise RuntimeError(f"reranker 回傳的分數數量不符：{scores.shape}")
                order = np.argsort(-scores, kind="stable")
                return np.concatenate([order.astype(np.float64), scores[order]])

            workers = reranker.concurrency
            chunk = max(16, workers * 4)
            ctx.progress(stage, cached, nq)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for start in range(cached, nq, chunk):
                    end = min(start + chunk, nq)
                    rows = list(pool.map(one, range(start, end))) if workers > 1 else [one(i) for i in range(start, end)]
                    store.append(start, np.vstack(rows))
                    ctx.progress(stage, end, nq)
    finally:
        if reranker is not None:
            reranker.close()
    data = store.load(nq)
    orders, scores = data[:, :n].astype(np.int64), data[:, n:]
    ranking = rerank_ranking(src_ranking, n, orders, scores)
    rid = save_result(
        db,
        ranking,
        {
            "job_id": ctx.job_id,
            "kind": "rerank",
            "dataset_key": ds.key,
            "label": cfg["label"],
            "model": cfg["params"]["model"],
            "provider": cfg["reranker"],
            "pair_label": source.get("pair_label"),
            "query_prefix": source.get("query_prefix"),
            "document_prefix": source.get("document_prefix"),
            "dims": source.get("dims"),
            "parent_result_id": source["id"],
            "num_documents": len(ds.documents),
            "config": {
                **cfg,
                "source": {k: source.get(k) for k in ("id", "label", "kind", "model", "provider", "pair_label", "dims")},
                "source_metrics": {k: v for k, v in (source.get("metrics") or {}).items() if k in ("hit@5", "mrr@5")},
            },
            "timing": {"seconds": round(time.time() - t0, 2), "cached": cached, "total": nq},
        },
    )
    return [rid]


# ---- 其他任務 ------------------------------------------------------------


def run_prepare(cfg: dict, ctx: JobContext) -> list[str]:
    cfg = normalize_prepare_config(cfg)
    prepare_dataset(cfg["dataset"], progress=lambda st, d, t: ctx.progress(st, d, t), force=cfg["force"])
    return []


def run_legacy_import(cfg: dict, ctx: JobContext) -> list[str]:
    from embedbench.legacy import import_legacy

    cfg = normalize_legacy_config(cfg)
    return import_legacy(cfg["path"], cfg["dataset"], cfg["models"], ctx)


RUNNERS: dict[str, Callable[[dict, JobContext], list[str]]] = {
    "embedding": run_embedding,
    "bm25": run_bm25,
    "rerank": run_rerank,
    "prepare_dataset": run_prepare,
    "legacy_import": run_legacy_import,
}

NORMALIZERS: dict[str, Callable[[dict], dict]] = {
    "embedding": normalize_embedding_config,
    "bm25": normalize_bm25_config,
    "rerank": normalize_rerank_config,
    "prepare_dataset": normalize_prepare_config,
    "legacy_import": normalize_legacy_config,
}


def job_title(kind: str, cfg: dict) -> str:
    match kind:
        case "embedding":
            return f"Embedding 評測：{cfg['label']}（{', '.join(cfg['datasets'])}）"
        case "bm25":
            return f"BM25 基準：{', '.join(cfg['datasets'])}"
        case "rerank":
            return f"Reranker 評測：{cfg['label']}"
        case "prepare_dataset":
            return f"下載資料集：{cfg['dataset']}"
        case "legacy_import":
            return f"匯入舊資料庫：{cfg['path']}"
    return kind


# ---- 連線測試 ------------------------------------------------------------

SAMPLE_QUERY = "臺灣最高的山是哪一座？"
SAMPLE_DOC_TITLE = "玉山"
SAMPLE_DOC = "玉山是臺灣最高峰，主峰海拔 3,952 公尺，也是東北亞第一高峰，位於南投縣、嘉義縣與高雄市交界。"
SAMPLE_NEGATIVE = "臺北 101 是位於臺北市信義區的摩天大樓，曾經是世界最高的建築物。"


def _sample(dataset_key: str | None):
    if dataset_key:
        try:
            ds = get_dataset(dataset_key, prepare=False)
            q = ds.eval_queries[0]
            doc = ds.documents[next(iter(ds.relevant(q.id)))]
            other = ds.documents[(next(iter(ds.relevant(q.id))) + len(ds.documents) // 2) % len(ds.documents)]
            return q.text, doc.title, doc.text, other.text
        except (FileNotFoundError, IndexError, StopIteration):
            pass
    return SAMPLE_QUERY, SAMPLE_DOC_TITLE, SAMPLE_DOC, SAMPLE_NEGATIVE


def probe_embedding(provider: str, params: dict, pairs: list[dict], dataset_key: str | None = None) -> dict:
    query, title, doc, _ = _sample(dataset_key)
    pairs = pairs or [{"label": "無前綴", "query": "", "document": ""}]
    t0 = time.time()
    p = create_provider(provider, params)
    load_seconds = time.time() - t0
    items, warnings = [], []
    try:
        for pair in pairs[:6]:
            q_text = prefixes.render(pair.get("query"), query)
            d_text = prefixes.render(pair.get("document"), doc, title)
            t1 = time.time()
            qv = p.embed([q_text], "query")[0]
            dv = p.embed([d_text], "document")[0]
            elapsed = time.time() - t1
            qn, dn = float(np.linalg.norm(qv)), float(np.linalg.norm(dv))
            cos = float(qv @ dv / (qn * dn)) if qn and dn else None
            nan = bool(np.isnan(qv).any() or np.isnan(dv).any())
            if nan:
                warnings.append(f"{pair.get('label')}：向量含有 NaN（若是 EmbeddingGemma，請確認沒有使用 float16）")
            if len(qv) != len(dv):
                warnings.append(f"{pair.get('label')}：查詢與文件向量維度不同")
            items.append(
                {
                    "label": pair.get("label") or pair_label(pair),
                    "query_text": q_text,
                    "document_text": d_text,
                    "dims": int(len(qv)),
                    "query_norm": qn,
                    "document_norm": dn,
                    "cosine": cos,
                    "nan": nan,
                    "elapsed_ms": round(elapsed * 1000),
                }
            )
        longest = _probe_longest_document(p, pairs[0], dataset_key, warnings)
    finally:
        p.close()
    return {"ok": True, "load_seconds": round(load_seconds, 2), "items": items, "warnings": warnings, "longest": longest}


def _probe_longest_document(p, pair: dict, dataset_key: str | None, warnings: list[str]) -> dict | None:
    """送出資料集中最長的文件，提早發現「超過模型長度上限」的問題。"""
    if not dataset_key:
        return None
    try:
        ds = get_dataset(dataset_key, prepare=False)
    except FileNotFoundError:
        return None
    if not ds.documents:
        return None
    template = pair.get("document")
    doc = max(ds.documents, key=lambda d: len(prefixes.render(template, d.text, d.title)))
    text = prefixes.render(template, doc.text, doc.title)
    p.pop_truncations()
    info = {"chars": len(text), "title": doc.title, "truncated": False, "kept_chars": len(text), "error": None}
    try:
        p.embed([text], "document")
    except ApiError as e:
        if not is_context_error(e):
            raise
        info["error"] = str(e)[:300]
        warnings.append(
            f"資料集中最長的文件（{len(text)} 字）超過模型長度上限，評測會失敗。"
            "請把參數「超過長度上限時」設為 auto（自動截短）。"
        )
        return info
    trunc = p.pop_truncations()
    info["notices"] = p.pop_notices()
    if trunc:
        info.update(truncated=True, kept_chars=trunc[0]["kept_chars"])
        warnings.append(
            f"資料集中最長的文件（{len(text)} 字）超過模型長度上限，評測時會自動從尾端截短"
            f"（這篇保留前 {trunc[0]['kept_chars']} 字）；被截短的筆數會記錄在結果中。"
        )
        warnings.extend(n for n in info["notices"] if n.startswith("無法載入 tokenizer"))
    return info


def probe_reranker(reranker: str, params: dict, dataset_key: str | None = None) -> dict:
    query, title, doc, negative = _sample(dataset_key)
    t0 = time.time()
    r = create_reranker(reranker, params)
    load_seconds = time.time() - t0
    try:
        t1 = time.time()
        scores = np.asarray(r.score(query, [doc, negative]), dtype=np.float64)
        elapsed = time.time() - t1
    finally:
        r.close()
    warnings = []
    if not scores[0] > scores[1]:
        warnings.append("相關文件的分數沒有高於無關文件，請確認模型與 API 設定")
    return {
        "ok": True,
        "load_seconds": round(load_seconds, 2),
        "query": query,
        "documents": [doc, negative],
        "scores": [float(s) if np.isfinite(s) else None for s in scores],
        "elapsed_ms": round(elapsed * 1000),
        "warnings": warnings,
    }
