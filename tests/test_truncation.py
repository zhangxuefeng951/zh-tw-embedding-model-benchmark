"""輸入超過模型長度上限時的自動截短。"""

import numpy as np
import pytest
from mock_server import fake_embedding

from embedbench.http import ApiError
from embedbench.jobs import run_job_sync
from embedbench.providers import create_provider
from embedbench.results import get_result, list_results
from embedbench.runner import probe_embedding
from embedbench.truncation import is_context_error, length_ratio, parse_limits

# 使用者實際遇到的錯誤（LiteLLM 轉發 vLLM）
LITELLM_BODY = (
    '{"error":{"message":"litellm.ContextWindowExceededError: litellm.BadRequestError: ContextWindowExceededError: '
    "OpenAIException - Error code: 400 - {'error': {'message': \\\"This model's maximum context length is 512 tokens. "
    "However, you requested 0 output tokens and your prompt contains at least 513 input tokens, for a total of at least "
    '513 tokens. Please reduce the length of the input prompt or the number of requested output tokens. '
    "(parameter=input_tokens, value=513)\\\", 'type': 'BadRequestError',"
)


def test_parse_context_errors():
    e = ApiError("x", 400, LITELLM_BODY)
    assert is_context_error(e)
    # vLLM 的「at least 513」只是下限，不能當成實際長度
    assert parse_limits(e) == (512, None)
    assert length_ratio(e) is None
    tei = ApiError("x", 413, '{"error":"Input validation error: `inputs` must have less than 512 tokens. Given: 1024"}')
    assert is_context_error(tei) and length_ratio(tei) == pytest.approx(0.5)
    openai = ApiError(
        "x", 400, "This model's maximum context length is 8192 tokens, however you requested 9000 tokens (9000 in your prompt; 0 for the completion)."
    )
    assert length_ratio(openai) == pytest.approx(8192 / 9000)
    assert not is_context_error(ApiError("x", 400, '{"error": "model not found"}'))
    assert not is_context_error(ApiError("x", 500, LITELLM_BODY))


@pytest.mark.parametrize("style", ["litellm", "tei", "openai"])
def test_auto_truncate_keeps_prefix_and_other_texts(mock_api, style):
    state, url = mock_api
    state.max_tokens, state.error_style = 40, style
    p = create_provider("openai", {"base_url": url, "model": "m", "batch_size": 8, "concurrency": 1})
    short = [f"passage: 短文{i}" for i in range(5)]
    long_text = "passage: " + "很長的段落內容" * 20
    texts = short[:2] + [long_text] + short[2:]
    vecs = p.embed(texts, "document")
    assert vecs.shape == (6, 256)
    trunc = p.pop_truncations()
    assert len(trunc) == 1 and trunc[0]["text"] == long_text
    assert trunc[0]["kept_chars"] <= 40
    # 截短的是尾端：送出的文字仍以前綴開頭，且向量就是截短後文字的向量
    sent = [t for t in state.embedding_inputs if t.startswith("passage: 很長") and len(t) <= 40]
    assert sent and long_text.startswith(sent[-1])
    assert np.allclose(vecs[2], fake_embedding(sent[-1]))
    # 其他文字照常計算
    for i, t in zip([0, 1, 3, 4, 5], texts[:2] + texts[3:]):
        assert np.allclose(vecs[i], fake_embedding(t), atol=1e-6)


@pytest.mark.parametrize("chars_per_token", [1.0, 1.6])
def test_truncate_much_longer_text_close_to_limit(mock_api, chars_per_token):
    """使用者遇到的情況：vLLM 只回報「at least 513」，而文件長度是上限的好幾倍。"""
    state, url = mock_api
    state.max_tokens, state.chars_per_token = 512, chars_per_token
    max_chars = int(512 * chars_per_token)
    p = create_provider("openai", {"base_url": url, "model": "e5", "batch_size": 4, "concurrency": 1})
    long_texts = [f"passage: 第{i}篇" + "很長的段落內容" * (150 + 20 * i) for i in range(12)]
    vecs = p.embed(long_texts, "document")
    assert vecs.shape == (12, 256)
    trunc = p.pop_truncations()
    assert len(trunc) == 12
    for t in trunc:
        assert t["kept_chars"] <= max_chars
        assert t["kept_chars"] >= max_chars * 0.97  # 接近上限，不會截掉太多
    # 學會「每個 token 幾個字元」之後，平均每篇只需要少量請求
    assert len(state.requests) / 12 < 8


def test_prefix_longer_than_limit_fails_cleanly(mock_api):
    state, url = mock_api
    state.max_tokens, state.chars_per_token = 1, 0.5  # 連 1 個字元（2 個 token）都超過上限
    p = create_provider("openai", {"base_url": url, "model": "m", "concurrency": 1})
    with pytest.raises(ApiError, match="截短後仍超過"):
        p.embed(["任何文字"], "document")
    assert len(state.requests) < 10


def test_error_mode_raises(mock_api):
    state, url = mock_api
    state.max_tokens = 10
    p = create_provider("openai", {"base_url": url, "model": "m", "truncate": "error"})
    with pytest.raises(ApiError, match="maximum context length"):
        p.embed(["這是一段超過十個字的文字內容"], "document")


def test_pipeline_records_truncation_and_cache(mock_api, tiny_dataset):
    state, url = mock_api
    longest = max(len("passage: " + d.text) for d in tiny_dataset.documents)
    state.max_tokens = 45
    n_long = sum(len("passage: " + d.text) > 45 for d in tiny_dataset.documents)
    assert 0 < n_long < len(tiny_dataset.documents) and longest > 45
    cfg = {
        "provider": "openai",
        "params": {"base_url": url, "model": "e5", "batch_size": 4},
        "prefix_pairs": [{"label": "e5", "query": "query: ", "document": "passage: "}],
        "datasets": [tiny_dataset.key],
    }
    out = run_job_sync("embedding", cfg)
    assert any("超過模型長度上限" in m for m in out["messages"])
    r = get_result(out["result_ids"][0])
    assert r["health"]["document"]["truncated"] == n_long
    assert r["health"]["query"]["truncated"] == 0
    # 第二次沿用快取：不再呼叫 API，但截斷筆數仍然正確
    state.embedding_inputs.clear()
    out2 = run_job_sync("embedding", cfg)
    assert state.embedding_inputs == []
    assert get_result(out2["result_ids"][0])["health"]["document"]["truncated"] == n_long
    rows = list_results(tiny_dataset.key)
    assert all(row["truncated_texts"]["document"] == n_long for row in rows)


def test_probe_reports_longest_document(mock_api, tiny_dataset):
    state, url = mock_api
    state.max_tokens = 45
    pairs = [{"label": "e5", "query": "query: ", "document": "passage: "}]
    out = probe_embedding("openai", {"base_url": url, "model": "e5"}, pairs, tiny_dataset.key)
    assert out["longest"]["truncated"] and out["longest"]["kept_chars"] <= 45
    assert any("自動從尾端截短" in w for w in out["warnings"])
    out = probe_embedding("openai", {"base_url": url, "model": "e5", "truncate": "error"}, pairs, tiny_dataset.key)
    assert out["longest"]["error"]
    assert any("評測會失敗" in w for w in out["warnings"])
    state.max_tokens = None
    out = probe_embedding("openai", {"base_url": url, "model": "e5"}, pairs, tiny_dataset.key)
    assert out["longest"] == {**out["longest"], "truncated": False, "error": None}
    assert not out["warnings"]
