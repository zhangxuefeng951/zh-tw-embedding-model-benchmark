"""輸入超過模型長度上限時的處理。

透過 API 呼叫時，系統拿不到模型的 tokenizer，因此採用「被拒絕後再截短」的做法：
1. 一批文字被拒絕時，對半拆開重送，找出真正過長的那幾筆（其他文字照常計算）
2. 過長的文字從尾端截短後重送（保留前綴與開頭，與 sentence-transformers 的預設截斷方式相同）；
   能從錯誤訊息解析出「上限 / 實際 token 數」時按比例截短，否則每次縮短 20%
"""

from __future__ import annotations

import re
from collections.abc import Callable

import numpy as np

from embedbench.http import ApiError

_CONTEXT_ERROR = re.compile(
    r"context length|context window|contextwindowexceeded|too long|token limit|must have less than"
    r"|maximum.{0,40}tokens|exceed.{0,40}(?:token|length)|input_tokens|max_model_len",
    re.IGNORECASE | re.DOTALL,
)

# 上限：vLLM / OpenAI「maximum context length is 512 tokens」、TEI「must have less than 512 tokens」
_LIMIT = [
    re.compile(r"maximum context length is (\d+)", re.I),
    re.compile(r"less than (\d+) tokens", re.I),
    re.compile(r"max(?:imum)?[_ ](?:model[_ ])?(?:input[_ ])?(?:len(?:gth)?|tokens?)\D{0,15}(\d+)", re.I),
]
# 實際長度：vLLM「contains at least 513 input tokens」、TEI「Given: 600」、OpenAI「resulted in 9000 tokens」
_ACTUAL = [
    re.compile(r"(?:contains at least|contains|at least)\s+(\d+)\s+(?:input\s+)?tokens", re.I),
    re.compile(r"Given:\s*(\d+)", re.I),
    re.compile(r"resulted in (\d+) tokens", re.I),
    re.compile(r"requested (\d+) tokens \(", re.I),
]

MAX_ATTEMPTS = 12


def is_context_error(e: BaseException) -> bool:
    if not isinstance(e, ApiError) or e.status not in (400, 413, 422):
        return False
    return bool(_CONTEXT_ERROR.search(e.body or str(e)))


def _first(patterns: list[re.Pattern], text: str) -> int | None:
    for p in patterns:
        m = p.search(text)
        if m and int(m.group(1)) > 0:
            return int(m.group(1))
    return None


def length_ratio(e: ApiError) -> float | None:
    """從錯誤訊息估計「上限 / 實際長度」，解析不到時回傳 None。"""
    text = e.body or str(e)
    limit, actual = _first(_LIMIT, text), _first(_ACTUAL, text)
    if limit and actual and actual > limit:
        return limit / actual
    return None


def embed_with_truncation(
    texts: list[str],
    request: Callable[[list[str]], np.ndarray],
    on_truncate: Callable[[str, str], None],
) -> np.ndarray:
    """呼叫 request(texts)；遇到長度錯誤時拆批並截短過長的文字。on_truncate(原文, 截短後) 用來記錄。"""
    try:
        return request(texts)
    except ApiError as e:
        if not is_context_error(e):
            raise
        if len(texts) > 1:
            mid = len(texts) // 2
            return np.vstack(
                [
                    embed_with_truncation(texts[:mid], request, on_truncate),
                    embed_with_truncation(texts[mid:], request, on_truncate),
                ]
            )
        return _truncate_one(texts[0], request, on_truncate, e)


def _truncate_one(
    original: str,
    request: Callable[[list[str]], np.ndarray],
    on_truncate: Callable[[str, str], None],
    error: ApiError,
) -> np.ndarray:
    text = original
    for _ in range(MAX_ATTEMPTS):
        ratio = length_ratio(error)
        new_len = int(len(text) * (ratio * 0.98 if ratio else 0.8))
        new_len = min(new_len, len(text) - 1)
        if new_len <= 0:
            break
        text = text[:new_len]
        try:
            out = request([text])
        except ApiError as e:
            if not is_context_error(e):
                raise
            error = e
            continue
        on_truncate(original, text)
        return out
    raise ApiError(f"文字截短 {MAX_ATTEMPTS} 次後仍超過模型長度上限：{error}", error.status, error.body)
