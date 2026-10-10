"""Web API 與頁面。"""

import io
import time

import pytest

from embedbench.web import create_app


@pytest.fixture
def client():
    app = create_app(start_jobs=True)
    yield app.test_client()
    app.extensions["jobs"].stop()


def wait_job(client, job_id, timeout=30):
    end = time.time() + timeout
    while time.time() < end:
        job = client.get(f"/api/jobs/{job_id}").get_json()
        if job["status"] not in ("queued", "running"):
            return job
        time.sleep(0.1)
    raise TimeoutError(job_id)


def test_pages_render(client):
    for path in ("/", "/new", "/jobs", "/datasets", "/presets", "/settings", "/results/x", "/results/x/queries"):
        assert client.get(path).status_code == 200, path


def test_presets_and_providers(client):
    presets = client.get("/api/presets?kind=embedding").get_json()
    by_id = {p["id"]: p for p in presets}
    gemma = by_id["embeddinggemma-2"]
    assert gemma["provider"] == "openai"
    queries = {p["query"] for p in gemma["prefix_pairs"]}
    assert {
        "task: search result | query: ",
        "task: question answering | query: ",
        "task: classification | query: ",
        "task: sentence similarity | query: ",
    } <= queries
    assert "title: none | text: " in {p["document"] for p in gemma["prefix_pairs"]}
    e5 = by_id["multilingual-e5-large"]["prefix_pairs"][0]
    assert (e5["query"], e5["document"]) == ("query: ", "passage: ")
    assert by_id["e5-large-v2"]["prefix_pairs"][0]["query"] == "query: "
    instruct = by_id["multilingual-e5-large-instruct"]["prefix_pairs"][0]
    assert instruct["query"].startswith("Instruct: ") and instruct["query"].endswith("\nQuery: ") and instruct["document"] == ""
    assert client.get("/api/presets?kind=reranker").get_json()
    providers = {p["type"] for p in client.get("/api/providers?kind=embedding").get_json()}
    assert {"openai", "cohere", "voyage", "gemini", "vertex", "sentence_transformers"} <= providers


def test_user_preset_roundtrip(client):
    preset = {"name": "我的模型", "provider": "openai", "params": {"model": "x"}, "prefix_pairs": [{"label": "a", "query": "q: ", "document": "", "default": True}]}
    saved = client.post("/api/presets", json={"kind": "embedding", "preset": preset}).get_json()
    assert saved["id"].startswith("user-") and saved["source"] == "user"
    yaml_text = client.get(f"/api/presets/{saved['id']}/yaml?kind=embedding").get_data(as_text=True)
    assert "我的模型" in yaml_text and "q: " in yaml_text
    assert client.delete(f"/api/presets/{saved['id']}").status_code == 200
    assert client.delete("/api/presets/embeddinggemma-2").status_code == 400


def test_upload_and_full_job_flow(client, mock_api):
    state, url = mock_api
    from conftest import tiny_rows

    lines = ["question,paragraph,title"] + [f"{r['question']},{r['paragraph']},{r['title']}" for r in tiny_rows()]
    resp = client.post(
        "/api/datasets/upload",
        data={"format": "simple", "name": "上傳測試", "file": (io.BytesIO("\n".join(lines).encode()), "t.csv")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200, resp.get_json()
    key = resp.get_json()["key"]
    sample = client.get(f"/api/datasets/{key}/sample?n=2").get_json()
    assert len(sample["items"]) == 2 and sample["has_titles"]

    cfg = {
        "provider": "openai",
        "params": {"base_url": url, "model": "m"},
        "prefix_pairs": [{"label": "p", "query": "query: ", "document": "passage: "}],
        "datasets": [key],
        "eval_dims": "256,64",
    }
    test = client.post("/api/test/embedding", json={**cfg, "dataset": key}).get_json()
    assert test["ok"] and test["items"][0]["query_text"].startswith("query: ")
    job = client.post("/api/jobs", json={"kind": "embedding", "config": cfg}).get_json()
    job = wait_job(client, job["id"])
    assert job["status"] == "done", job
    assert len(job["result_ids"]) == 2

    rows = client.get(f"/api/results?dataset={key}&ks=1,7").get_json()
    assert len(rows) == 2 and "hit@7" in rows[0]["metrics"]
    rid = rows[0]["id"]
    detail = client.get(f"/api/results/{rid}?ks=1,5").get_json()
    assert detail["query_prefix"] == "query: "
    curve = client.get(f"/api/results/{rid}/curve").get_json()
    assert len(curve["hit"]) == 20 and curve["hit"] == sorted(curve["hit"])
    q = client.get(f"/api/results/{rid}/queries?k=1&status=all").get_json()
    assert q["num_queries"] == 16
    md = client.get(f"/api/export/leaderboard?dataset={key}&format=md&ks=5").get_data(as_text=True)
    assert "Hit@5" in md
    assert client.get(f"/api/export/results/{rid}?format=queries_csv").status_code == 200

    rr = client.post(
        "/api/jobs",
        json={"kind": "rerank", "config": {"source_result_id": rid, "reranker": "rerank_api", "params": {"base_url": url, "model": "r"}, "candidates": 5}},
    ).get_json()
    assert wait_job(client, rr["id"])["status"] == "done"
    assert client.patch(f"/api/results/{rid}", json={"label": "新名稱"}).status_code == 200
    assert client.delete(f"/api/results/{rid}").status_code == 200
    assert client.get(f"/api/results/{rid}").status_code == 404


def test_validation_errors(client):
    bad = client.post("/api/jobs", json={"kind": "embedding", "config": {"provider": "openai", "params": {}, "datasets": ["drcd-test"]}})
    assert bad.status_code == 400 and "模型名稱" in bad.get_json()["error"]
    bad = client.post("/api/jobs", json={"kind": "embedding", "config": {"params": {"model": "m"}, "datasets": ["nope"]}})
    assert bad.status_code == 400
    bad = client.post("/api/jobs", json={"kind": "embedding", "config": {"params": {"model": "m", "batch_size": "abc"}, "datasets": ["drcd-test"]}})
    assert bad.status_code == 400 and "批次大小" in bad.get_json()["error"]
    assert client.put("/api/settings", json={"max_concurrent_jobs": 0}).status_code == 400
    ok = client.put("/api/settings", json={"max_concurrent_jobs": 2, "default_ks": "1,5,10"}).get_json()
    assert ok["max_concurrent_jobs"] == 2 and ok["default_ks"] == [1, 5, 10]


def test_job_failure_is_reported(client, tiny_dataset):
    job = client.post(
        "/api/jobs",
        json={"kind": "embedding", "config": {"params": {"model": "m", "base_url": "http://127.0.0.1:9/v1", "max_retries": 0, "timeout": 2}, "datasets": [tiny_dataset.key]}},
    ).get_json()
    job = wait_job(client, job["id"], timeout=60)
    assert job["status"] == "failed"
    assert job["message"]
