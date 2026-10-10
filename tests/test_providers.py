"""Provider / reranker 的請求格式。"""

import numpy as np
import pytest

from embedbench.providers import create_provider, provider_signature
from embedbench.providers.api import join_url
from embedbench.rerankers import create_reranker
from embedbench.rerankers.api import parse_ranking


def test_join_url():
    assert join_url("http://h:8000/v1", "embeddings") == "http://h:8000/v1/embeddings"
    assert join_url("http://h:8000/v1/", "embeddings") == "http://h:8000/v1/embeddings"
    assert join_url("http://h:8000/v1/embeddings", "embeddings") == "http://h:8000/v1/embeddings"


def test_openai_provider_batches_and_extra(mock_api):
    state, url = mock_api
    p = create_provider(
        "openai",
        {"base_url": url, "model": "m", "batch_size": 2, "concurrency": 3, "dimensions": 64, "query_extra": {"task": "q"}},
    )
    texts = [f"文字{i}" for i in range(5)]
    vecs = p.embed(texts, "query")
    assert vecs.shape == (5, 64)
    bodies = [r["body"] for r in state.requests]
    assert sorted(len(b["input"]) for b in bodies) == [1, 2, 2]
    assert all(b["dimensions"] == 64 and b["task"] == "q" and b["model"] == "m" for b in bodies)
    # 順序要與輸入一致
    single = create_provider("openai", {"base_url": url, "model": "m", "dimensions": 64}).embed(["文字3"], "query")
    assert np.allclose(vecs[3], single[0])


def test_openai_provider_retries(mock_api):
    state, url = mock_api
    state.fail_next = 2
    p = create_provider("openai", {"base_url": url, "model": "m", "max_retries": 3})
    import embedbench.http as http

    orig = http._sleep
    http._sleep = lambda *a: None
    try:
        assert p.embed(["a"], "document").shape[0] == 1
    finally:
        http._sleep = orig
    assert len(state.requests) == 3


def test_api_key_from_env(mock_api, monkeypatch):
    state, url = mock_api
    p = create_provider("openai", {"base_url": url, "model": "m", "api_key_env": "MY_TEST_KEY"})
    with pytest.raises(ValueError, match="MY_TEST_KEY"):
        p.embed(["a"], "query")
    monkeypatch.setenv("MY_TEST_KEY", "secret")
    p.embed(["a"], "query")
    assert state.requests[-1]["auth"] == "Bearer secret"


def test_signature_ignores_runtime_params():
    a = provider_signature("openai", {"model": "m", "base_url": "u", "batch_size": 8, "api_key_env": "A"})
    b = provider_signature("openai", {"model": "m", "base_url": "u", "batch_size": 64, "concurrency": 9})
    c = provider_signature("openai", {"model": "m", "base_url": "u", "dimensions": 256})
    assert a == b
    assert a != c


@pytest.fixture
def capture(monkeypatch):
    calls = []

    def install(module, response):
        def fake(url, body, headers=None, **kw):
            calls.append({"url": url, "body": body, "headers": headers})
            return response(body) if callable(response) else response

        monkeypatch.setattr(module, "post_json", fake)
        return calls

    return install


def test_cohere_voyage_gemini_vertex_formats(capture, monkeypatch):
    import embedbench.providers.api as api

    monkeypatch.setenv("K", "key")
    calls = capture(api, lambda body: {"embeddings": {"float": [[1.0, 0.0]] * len(body["texts"])}})
    create_provider("cohere", {"model": "embed-v4.0", "api_key_env": "K"}).embed(["a", "b"], "query")
    assert calls[-1]["url"].endswith("/v2/embed")
    assert calls[-1]["body"]["input_type"] == "search_query"

    calls = capture(api, lambda body: {"data": [{"index": i, "embedding": [1.0, 0.0]} for i in range(len(body.get("input") or body.get("inputs")))]})
    create_provider("voyage", {"model": "voyage-3.5", "api_key_env": "K"}).embed(["a"], "document")
    assert calls[-1]["body"]["input_type"] == "document" and calls[-1]["url"].endswith("/v1/embeddings")
    create_provider("voyage", {"model": "voyage-multimodal-3", "endpoint": "multimodalembeddings", "api_key_env": "K"}).embed(["a"], "query")
    assert calls[-1]["body"]["inputs"] == [{"content": [{"type": "text", "text": "a"}]}]

    calls = capture(api, lambda body: {"embeddings": [{"values": [1.0, 0.0]} for _ in body["requests"]]})
    create_provider("gemini", {"model": "gemini-embedding-001", "api_key_env": "K", "dimensions": 768}).embed(["a"], "document")
    req = calls[-1]["body"]["requests"][0]
    assert req["taskType"] == "RETRIEVAL_DOCUMENT" and req["outputDimensionality"] == 768
    assert calls[-1]["url"].endswith("models/gemini-embedding-001:batchEmbedContents")
    create_provider("gemini", {"model": "gemini-embedding-001", "api_key_env": "K", "query_task_type": ""}).embed(["a"], "query")
    assert "taskType" not in calls[-1]["body"]["requests"][0]

    monkeypatch.setenv("GCP_ACCESS_TOKEN", "tok")
    calls = capture(api, lambda body: {"predictions": [{"embeddings": {"values": [1.0, 0.0]}} for _ in body["instances"]]})
    create_provider("vertex", {"model": "text-multilingual-embedding-002", "project": "p"}).embed(["a"], "query")
    assert "projects/p/locations/us-central1" in calls[-1]["url"]
    assert calls[-1]["body"]["instances"][0]["task_type"] == "RETRIEVAL_QUERY"
    assert calls[-1]["headers"]["Authorization"] == "Bearer tok"


def test_rerank_api_and_llm(mock_api):
    state, url = mock_api
    docs = ["玉山是臺灣最高峰", "臺北101是摩天大樓", "日月潭在南投"]
    scores = create_reranker("rerank_api", {"base_url": url, "model": "r"}).score("臺灣最高的山", docs)
    assert scores.argmax() == 0
    llm = create_reranker("llm_listwise", {"base_url": url, "model": "gpt", "api_key_env": "", "window_size": 2, "step": 1})
    scores = llm.score("臺灣最高的山", docs)
    assert scores.argmax() == 0
    assert sorted(scores.tolist()) == [1.0, 2.0, 3.0]


def test_parse_ranking():
    assert parse_ranking('{"ranking": [3, 1, 2]}') == [3, 1, 2]
    assert parse_ranking("[2,1]") == [2, 1]
    assert parse_ranking("[2] > [1] > [3]") == [2, 1, 3]


def test_local_provider_without_packages_has_clear_error():
    import importlib.util

    if importlib.util.find_spec("sentence_transformers"):
        pytest.skip("已安裝本地模型套件")
    with pytest.raises(RuntimeError, match="uv sync --extra local"):
        create_provider("sentence_transformers", {"model": "x"})
