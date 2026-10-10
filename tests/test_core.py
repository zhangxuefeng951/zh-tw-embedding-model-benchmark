"""前綴、指標、BM25、資料集解析。"""

import io
import json
import zipfile

import numpy as np
import pytest

from embedbench import prefixes
from embedbench.bm25 import BM25Index, reference_scores
from embedbench.datasets.base import parse_qrels_lines
from embedbench.datasets.custom import parse_beir, parse_beir_zip, parse_simple_file
from embedbench.ranking import compute_metrics, rank_by_scores, rank_dense, rerank_ranking


# ---- 前綴 -------------------------------------------------------------------


def test_render_plain_prefix():
    assert prefixes.render("query: ", "台灣最高的山？") == "query: 台灣最高的山？"
    assert prefixes.render("", "文字") == "文字"
    assert prefixes.render(None, "文字") == "文字"


def test_render_embeddinggemma_prefixes():
    assert prefixes.render("task: search result | query: ", "問題") == "task: search result | query: 問題"
    assert prefixes.render("title: none | text: ", "段落", "玉山") == "title: none | text: 段落"
    assert prefixes.render("title: {title} | text: ", "段落", "玉山") == "title: 玉山 | text: 段落"
    assert prefixes.render("title: {title} | text: ", "段落", "") == "title: none | text: 段落"
    assert prefixes.render("title: {title} | text: ", "段落", None) == "title: none | text: 段落"


def test_render_text_placeholder_and_newline():
    assert prefixes.render("<q>{text}</q>", "內容") == "<q>內容</q>"
    instruct = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
    assert prefixes.render(instruct, "問題") == instruct + "問題"


def test_render_does_not_expand_braces_in_text():
    assert prefixes.render("title: {title} | text: ", "含有 {title} 的文字", "標題") == "title: 標題 | text: 含有 {title} 的文字"


# ---- 指標 -------------------------------------------------------------------


def brute_metrics(scores: np.ndarray, relevant: list[dict[int, float]], k: int) -> dict:
    hit, mrr, ndcg, recall = [], [], [], []
    for row, rel in zip(scores, relevant):
        order = list(np.argsort(-row, kind="stable"))
        top = order[:k]
        ranks = [order.index(d) + 1 for d in rel]
        best = min(ranks)
        hit.append(best <= k)
        mrr.append(1 / best if best <= k else 0)
        recall.append(sum(1 for d in rel if d in top) / len(rel))
        dcg = sum(g / np.log2(order.index(d) + 2) for d, g in rel.items() if order.index(d) < k)
        ideal = sorted(rel.values(), reverse=True)[:k]
        idcg = sum(g / np.log2(i + 2) for i, g in enumerate(ideal))
        ndcg.append(dcg / idcg)
    return {f"hit@{k}": np.mean(hit), f"mrr@{k}": np.mean(mrr), f"ndcg@{k}": np.mean(ndcg), f"recall@{k}": np.mean(recall)}


def test_metrics_match_brute_force():
    rng = np.random.default_rng(0)
    nq, nd = 40, 300
    scores = rng.normal(size=(nq, nd)).astype(np.float32)
    relevant = []
    for _ in range(nq):
        docs = rng.choice(nd, size=rng.integers(1, 4), replace=False)
        relevant.append({int(d): float(rng.integers(1, 3)) for d in docs})
    r = rank_by_scores(lambda s, e: scores[s:e], nq, nd, relevant, block=7)
    m = compute_metrics(r, [1, 3, 5, 10, 50])
    for k in [1, 3, 5, 10, 50]:
        expected = brute_metrics(scores, relevant, k)
        for key, value in expected.items():
            assert m[key] == pytest.approx(value), key
    # 前 T 名與完整排序一致
    order = np.argsort(-scores, axis=1, kind="stable")[:, :100]
    assert np.array_equal(r.top_idx, order)


def test_single_relevant_matches_old_benchmark_definition():
    """舊版 benchmark.py：Hit@5 = 正解在前 5 名、MRR@5 = 1/(名次)。"""
    scores = np.array([[0.9, 0.8, 0.1], [0.1, 0.2, 0.3], [0.5, 0.4, 0.6]], dtype=np.float32)
    relevant = [{0: 1.0}, {0: 1.0}, {1: 1.0}]
    r = rank_by_scores(lambda s, e: scores[s:e], 3, 3, relevant)
    m = compute_metrics(r, [1, 5])
    assert m["hit@1"] == pytest.approx(1 / 3)
    assert m["mrr@5"] == pytest.approx((1 + 1 / 3 + 1 / 3) / 3)


def test_rank_dense_truncation_renormalizes():
    q = np.array([[1.0, 0.0, 5.0]], dtype=np.float32)
    d = np.array([[1.0, 0.0, -5.0], [0.0, 1.0, 5.0]], dtype=np.float32)
    full = rank_dense(q, d, [{0: 1.0}])
    trunc = rank_dense(q, d, [{0: 1.0}], dims=2)
    assert full.rel_rank[0] == 2  # 完整維度時文件 1 比較近
    assert trunc.rel_rank[0] == 1  # 只看前 2 維時文件 0 最近


def test_rerank_ranking_keeps_outside_candidates():
    scores = np.array([[0.9, 0.8, 0.7, 0.6, 0.5]], dtype=np.float32)
    src = rank_by_scores(lambda s, e: scores[s:e], 1, 5, [{3: 1.0}])
    assert src.rel_rank[0] == 4
    # 只重排前 3 名 → 正解（第 4 名）不在候選內，名次不變
    r = rerank_ranking(src, 3, np.array([[2, 1, 0]]), np.array([[3.0, 2.0, 1.0]]))
    assert r.rel_rank[0] == 4
    assert list(r.top_idx[0][:3]) == [2, 1, 0]
    # 重排前 4 名並把正解排到第一
    r2 = rerank_ranking(src, 4, np.array([[3, 0, 1, 2]]), np.array([[9.0, 3.0, 2.0, 1.0]]))
    assert r2.rel_rank[0] == 1
    assert compute_metrics(r2, [1])["hit@1"] == 1.0


# ---- BM25 -------------------------------------------------------------------


def test_bm25_matches_rank_bm25_formula():
    corpus = [["台灣", "最高", "的", "山", "是", "玉山"], ["日月潭", "在", "南投"], ["玉山", "在", "南投", "與", "嘉義"], ["的", "的", "的"]]
    index = BM25Index.build(corpus)
    for query in (["玉山", "南投"], ["的"], ["不存在"], ["玉山", "玉山"]):
        ours = index.scores(index.query_matrix([query]))[0]
        assert np.allclose(ours, reference_scores(corpus, query), atol=1e-5)


# ---- 資料集解析 --------------------------------------------------------------


def test_simple_csv_merges_same_paragraph():
    content = "question,paragraph,title\n問一,段落甲,標題\n問二,段落甲,標題\n問三,段落乙,\n".encode("utf-8-sig")
    ds = parse_simple_file(content, "x.csv", "custom-x", "x")
    assert len(ds.documents) == 2
    assert len(ds.eval_queries) == 3
    assert ds.relevant(ds.queries[0].id) == ds.relevant(ds.queries[1].id)
    assert ds.has_titles


def test_simple_jsonl_and_aliases():
    lines = [json.dumps({"query": "問", "context": "段落"}, ensure_ascii=False)]
    ds = parse_simple_file("\n".join(lines).encode(), "x.jsonl", "k", "n")
    assert ds.queries[0].text == "問" and ds.documents[0].text == "段落"
    with pytest.raises(ValueError):
        parse_simple_file(b"question,foo\nq,x\n", "x.csv", "k", "n")


def test_beir_files_and_zip():
    corpus = "\n".join(
        json.dumps(o, ensure_ascii=False) for o in [{"_id": "d1", "title": "T", "text": "甲"}, {"_id": "d2", "text": "乙"}]
    ).encode()
    queries = "\n".join(json.dumps(o, ensure_ascii=False) for o in [{"_id": "q1", "text": "問一"}, {"_id": "q2", "text": "沒有正解"}]).encode()
    qrels = b"query-id\tcorpus-id\tscore\nq1\td1\t2\nq1\td2\t1\n"
    ds = parse_beir(corpus, queries, qrels, "k", "n")
    assert [q.id for q in ds.eval_queries] == ["q1"]
    assert ds.relevant("q1") == {0: 2.0, 1: 1.0}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("set/corpus.jsonl", corpus)
        zf.writestr("set/queries.jsonl", queries)
        zf.writestr("set/qrels/test.tsv", qrels)
    ds2 = parse_beir_zip(buf.getvalue(), "k", "n")
    assert ds2.qrels == ds.qrels


def test_qrels_formats():
    assert parse_qrels_lines(["q1 Q0 d1 1", "q1 0 d2 0"]) == {"q1": {"d1": 1, "d2": 0}}
    assert parse_qrels_lines(["query-id\tcorpus-id\tscore", "q1\td1\t1"]) == {"q1": {"d1": 1}}
