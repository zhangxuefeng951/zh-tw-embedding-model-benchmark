"""完整流程：embedding 評測、快取續跑、BM25、reranker、舊資料匯入、匯出。"""

import json
import sqlite3

import numpy as np
import pytest

from embedbench.jobs import run_job_sync
from embedbench.results import get_result, list_results, load_ranking, query_details
from embedbench.runner import JobContext, probe_embedding, run_embedding

GEMMA_PAIRS = [
    {"label": "search", "query": "task: search result | query: ", "document": "title: none | text: "},
    {"label": "qa", "query": "task: question answering | query: ", "document": "title: none | text: "},
    {"label": "search+title", "query": "task: search result | query: ", "document": "title: {title} | text: "},
]


def config(url, dataset, **kw):
    return {
        "provider": "openai",
        "params": {"base_url": url, "model": "google/embeddinggemma-2", "batch_size": 4},
        "prefix_pairs": GEMMA_PAIRS,
        "datasets": [dataset],
        **kw,
    }


def test_embedding_run_applies_prefixes_and_shares_document_vectors(mock_api, tiny_dataset):
    state, url = mock_api
    out = run_job_sync("embedding", config(url, tiny_dataset.key, eval_dims=[256, 128]))
    assert len(out["result_ids"]) == 3 * 2  # 3 組前綴 × 2 種維度
    inputs = state.embedding_inputs
    n_docs, n_q = len(tiny_dataset.documents), len(tiny_dataset.eval_queries)
    # 兩種文件前綴各算一次（search 與 qa 共用 title: none）、兩種查詢前綴各算一次
    assert len(inputs) == 2 * n_docs + 2 * n_q
    assert sum(t.startswith("title: none | text: ") for t in inputs) == n_docs
    assert "title: 玉山 | text: 玉山是臺灣最高峰，主峰海拔三千九百五十二公尺，位於南投縣與嘉義縣交界。" in inputs
    assert sum(t.startswith("task: question answering | query: ") for t in inputs) == n_q
    rows = list_results(tiny_dataset.key, ks=[1, 5])
    assert {r["pair_label"] for r in rows} == {"search", "qa", "search+title"}
    assert {r["dims"] for r in rows} == {256, 128}
    for r in rows:
        assert 0 < r["metrics"]["hit@5"] <= 1
        assert r["metrics"]["hit@1"] <= r["metrics"]["hit@5"]
    full = [r for r in rows if r["dims"] == 256 and r["pair_label"] == "search"][0]
    assert not full["truncated"]
    detail = get_result(full["id"])
    assert detail["health"]["document"]["nan_rows"] == 0
    assert detail["config"]["params"]["model"] == "google/embeddinggemma-2"


def test_cache_resume_after_query_limit(mock_api, tiny_dataset):
    state, url = mock_api
    cfg = config(url, tiny_dataset.key, prefix_pairs=GEMMA_PAIRS[:1])
    run_job_sync("embedding", {**cfg, "query_limit": 5})
    n_docs = len(tiny_dataset.documents)
    assert len(state.embedding_inputs) == n_docs + 5
    state.embedding_inputs.clear()
    run_job_sync("embedding", cfg)
    # 文件全部沿用、查詢只補算剩下的
    assert len(state.embedding_inputs) == len(tiny_dataset.eval_queries) - 5
    state.embedding_inputs.clear()
    run_job_sync("embedding", {**cfg, "params": {**cfg["params"], "batch_size": 7, "concurrency": 1}})
    assert state.embedding_inputs == []  # 只改批次大小不影響快取
    rows = list_results(tiny_dataset.key)
    assert sorted(r["num_queries"] for r in rows) == [5, 16, 16]


def test_cancel_keeps_partial_cache(mock_api, tiny_dataset):
    state, url = mock_api
    from embedbench.runner import Cancelled

    calls = {"n": 0}

    def progress(p):
        calls["n"] += 1
        # 文件向量算完並存入快取後，在查詢階段取消
        if p.get("done", 0) >= 4 and "查詢" in p.get("stage", ""):
            ctx.cancel_event.set()

    ctx = JobContext("t", None, progress)
    cfg = config(url, tiny_dataset.key, prefix_pairs=GEMMA_PAIRS[:1])
    cfg["params"] = {**cfg["params"], "batch_size": 2, "concurrency": 1}
    with pytest.raises(Cancelled):
        run_embedding(cfg, ctx)
    done_before = len(state.embedding_inputs)
    state.embedding_inputs.clear()
    run_embedding(cfg, JobContext())
    total = len(tiny_dataset.documents) + len(tiny_dataset.eval_queries)
    assert done_before < total
    # 第二次只需要算查詢，文件向量沿用快取
    assert len(state.embedding_inputs) == len(tiny_dataset.eval_queries)


def test_bm25_and_rerank(mock_api, tiny_dataset):
    state, url = mock_api
    bm25 = run_job_sync("bm25", {"datasets": [tiny_dataset.key]})["result_ids"][0]
    emb = run_job_sync("embedding", config(url, tiny_dataset.key, prefix_pairs=GEMMA_PAIRS[:1]))["result_ids"][0]
    rr = run_job_sync(
        "rerank",
        {"source_result_id": emb, "reranker": "rerank_api", "params": {"base_url": url, "model": "r"}, "candidates": 5},
    )["result_ids"][0]
    r = get_result(rr)
    assert r["kind"] == "rerank" and r["parent_result_id"] == emb
    src, out = load_ranking(emb), load_ranking(rr)
    # 只重排前 5 名：Hit@5 以外的正解名次不變
    outside = src.rel_rank > 5
    assert np.array_equal(src.rel_rank[outside], out.rel_rank[outside])
    assert get_result(bm25)["metrics"]["hit@10"] > 0.5
    # 以 BM25 為第一階段也可以
    llm = run_job_sync(
        "rerank",
        {"source_result_id": bm25, "reranker": "llm_listwise", "params": {"base_url": url, "model": "gpt", "api_key_env": ""}, "candidates": 4},
    )["result_ids"][0]
    m = {r["id"]: r["metrics"] for r in list_results(tiny_dataset.key, ks=[4])}
    assert m[llm]["hit@4"] == pytest.approx(m[bm25]["hit@4"])  # 只重排前 4 名，Hit@4 不變


def test_query_details_and_export(mock_api, tiny_dataset):
    from embedbench import export

    state, url = mock_api
    rid = run_job_sync("embedding", config(url, tiny_dataset.key, prefix_pairs=GEMMA_PAIRS[:1]))["result_ids"][0]
    d = query_details(rid, k=1, status="miss", per_page=100)
    assert d["total"] == d["misses"]
    assert all(not item["hit"] for item in d["items"])
    all_items = query_details(rid, k=1, status="all", per_page=100)
    assert all_items["total"] == len(tiny_dataset.eval_queries)
    item = all_items["items"][0]
    assert item["gold"][0]["title"] and len(item["top"]) == 5
    rows = list_results(tiny_dataset.key, ks=[1, 5])
    md = export.leaderboard_markdown(rows, [1, 5])
    assert "| Hit@1 | Hit@5 | MRR@1 | MRR@5 |" in md
    csv_text = export.leaderboard_csv(rows, [1, 5])
    assert csv_text.startswith("﻿") and "hit@5" in csv_text
    assert "正解最佳名次" in export.result_queries_csv(rid)
    assert json.loads(export.result_json(rid, [1, 5]))["id"] == rid


def test_connection_test(mock_api, tiny_dataset):
    state, url = mock_api
    out = probe_embedding("openai", {"base_url": url, "model": "m"}, GEMMA_PAIRS[2:], tiny_dataset.key)
    item = out["items"][0]
    assert item["document_text"].startswith("title: ")
    assert item["dims"] == 256 and not item["nan"]


def test_legacy_import(tmp_path, tiny_dataset):
    from embedbench.legacy import import_legacy, list_legacy_models
    from tests_helpers import fake_vec

    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE paragraphs (id INTEGER PRIMARY KEY AUTOINCREMENT, content TEXT, embedding TEXT, model TEXT)")
    conn.execute(
        "CREATE TABLE questions (id INTEGER PRIMARY KEY AUTOINCREMENT, dataset_id INTEGER, content TEXT, embedding TEXT, model TEXT, paragraph_id INTEGER)"
    )
    for model in ("old-a", "old-b"):
        pid_of = {}
        for d in tiny_dataset.documents:
            cur = conn.execute("INSERT INTO paragraphs (content, embedding, model) VALUES (?, ?, ?)", (d.text, json.dumps(fake_vec(d.text).tolist()), model))
            pid_of[d.id] = cur.lastrowid
        for i, q in enumerate(tiny_dataset.eval_queries[: 10 if model == "old-b" else None]):
            gold = next(iter(tiny_dataset.qrels[q.id]))
            vec = fake_vec(q.text).tolist()
            raw = json.dumps([vec]) if model == "old-b" else json.dumps(vec)  # 舊 jina v4 存成二維
            conn.execute("INSERT INTO questions (dataset_id, content, embedding, model, paragraph_id) VALUES (?, ?, ?, ?, ?)", (i, q.text, raw, model, pid_of[gold]))
    conn.commit()
    conn.close()

    assert [m["model"] for m in list_legacy_models(str(db))] == ["old-a", "old-b"]
    ids = import_legacy(str(db), tiny_dataset.key, None, JobContext())
    assert len(ids) == 2
    a, b = (get_result(i) for i in ids)
    assert a["kind"] == "legacy" and a["num_queries"] == len(tiny_dataset.eval_queries)
    assert b["num_queries"] == 10
    assert load_ranking(b["id"]).query_index is None or len(load_ranking(b["id"]).query_index) == 10
    # 與直接計算一致
    from embedbench.ranking import compute_metrics, rank_dense

    qv = np.vstack([fake_vec(q.text) for q in tiny_dataset.eval_queries])
    dv = np.vstack([fake_vec(d.text) for d in tiny_dataset.documents])
    direct = compute_metrics(rank_dense(qv, dv, [tiny_dataset.relevant(q.id) for q in tiny_dataset.eval_queries]), [5])
    assert a["metrics"]["hit@5"] == pytest.approx(direct["hit@5"])
    # 重複匯入會取代舊結果
    import_legacy(str(db), tiny_dataset.key, ["old-a"], JobContext())
    assert len([r for r in list_results(tiny_dataset.key) if r["model"] == "old-a"]) == 1
    # 逐題分析在部分題目的匯入結果上也能用
    assert query_details(b["id"], per_page=50)["total"] == 10
