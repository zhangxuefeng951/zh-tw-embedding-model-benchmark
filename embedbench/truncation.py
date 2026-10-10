"""輸入超過模型長度上限時的處理。

透過 API 呼叫時，系統拿不到模型的 tokenizer，因此採用「被拒絕後再截短」的做法：
1. 一批文字被拒絕時，對半拆開重送，找出真正過長的那幾筆（其他文字照常計算）
2. 過長的文字從尾端截短（保留前綴與開頭，與 sentence-transformers 的預設截斷方式相同），
   以二分搜尋找出服務端能接受的最長開頭（誤差約 2%）

注意：vLLM 對過長的輸入只回報「at least 上限+1 個 token」，這是下限而不是實際長度，
不能拿來推算要截掉多少；只有 TEI（Given: N）等回報實際長度的訊息才拿來當第一次的猜測。
另外會從已截短成功的文字學習「每個 token 約幾個字元」，讓之後的文字更快找到長度。
"""

from __future__ import annotations

import re
import statistics
import threading
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
# 實際長度（只收「確切數字」）：TEI「Given: 600」、OpenAI「resulted in 9000 tokens」「requested 9000 tokens (」
_ACTUAL_EXACT = [
    re.compile(r"Given:\s*(\d+)", re.I),
    re.compile(r"resulted in (\d+) tokens", re.I),
    re.compile(r"requested (\d+) tokens \(", re.I),
    re.compile(r"prompt contains (\d+) (?:input )?tokens", re.I),  # 沒有「at least」時才是確切數字
]

MAX_ATTEMPTS = 30
TOLERANCE = 0.02  # 找到的長度與最大可接受長度的差距上限（比例）


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


def parse_limits(e: ApiError) -> tuple[int | None, int | None]:
    """從錯誤訊息取出（上限, 確切的實際長度）；「at least」之類的下限不算確切長度。"""
    text = e.body or str(e)
    limit = _first(_LIMIT, text)
    actual = _first(_ACTUAL_EXACT, text)
    if actual is not None and limit is not None and actual <= limit:
        actual = None
    return limit, actual


def length_ratio(e: ApiError) -> float | None:
    """從錯誤訊息估計「上限 / 實際長度」；只有確切的實際長度才計算，否則回傳 None。"""
    limit, actual = parse_limits(e)
    return limit / actual if limit and actual else None


class TruncationState:
    """同一個 provider 共用：從截短成功的文字學習「每個 token 約幾個字元」。"""

    def __init__(self):
        self._lock = threading.Lock()
        self._chars_per_token: list[float] = []

    def learn(self, kept_chars: int, limit: int | None) -> None:
        if limit:
            with self._lock:
                self._chars_per_token.append(kept_chars / limit)

    def estimate(self) -> float | None:
        with self._lock:
            return statistics.median(self._chars_per_token) if self._chars_per_token else None


def embed_with_truncation(
    texts: list[str],
    request: Callable[[list[str]], np.ndarray],
    on_truncate: Callable[[str, str], None],
    state: TruncationState | None = None,
) -> np.ndarray:
    """呼叫 request(texts)；遇到長度錯誤時拆批並截短過長的文字。on_truncate(原文, 截短後) 用來記錄。"""
    state = state or TruncationState()
    try:
        return request(texts)
    except ApiError as e:
        if not is_context_error(e):
            raise
        if len(texts) > 1:
            mid = len(texts) // 2
            return np.vstack(
                [
                    embed_with_truncation(texts[:mid], request, on_truncate, state),
                    embed_with_truncation(texts[mid:], request, on_truncate, state),
                ]
            )
        return _truncate_one(texts[0], request, on_truncate, e, state)


def _initial_guess(length: int, error: ApiError, state: TruncationState) -> int:
    limit, actual = parse_limits(error)
    if limit and actual:
        return int(length * limit / actual * 0.98)
    cpt = state.estimate()
    if limit and cpt:
        return int(limit * cpt * 0.98)
    return length // 2


def _truncate_one(
    original: str,
    request: Callable[[list[str]], np.ndarray],
    on_truncate: Callable[[str, str], None],
    error: ApiError,
    state: TruncationState,
) -> np.ndarray:
    """二分搜尋最長可接受的開頭：lo＝已知可接受的長度、hi＝已知會被拒絕的長度。"""
    limit, _ = parse_limits(error)
    lo, lo_vec, hi = 0, None, len(original)
    guess = _initial_guess(hi, error, state)
    for _ in range(MAX_ATTEMPTS):
        guess = max(1, min(guess, hi - 1))
        if guess <= lo or hi <= 1:
            break
        try:
            vec = request([original[:guess]])
        except ApiError as e:
            if not is_context_error(e):
                raise
            hi, error = guess, e
        else:
            lo, lo_vec = guess, vec
        if lo_vec is not None and hi - lo <= max(4, int(lo * TOLERANCE)):
            break
        if lo_vec is None:
            guess = hi // 2
        else:
            # 先往上試 3%（猜測準確時很快收斂），再在可接受與被拒絕之間二分
            guess = min((lo + hi) // 2, int(lo * 1.03) + 1)
    if lo_vec is None:
        raise ApiError(f"文字截短後仍超過模型長度上限：{error}", error.status, error.body)
    state.learn(lo, limit)
    on_truncate(original, original[:lo])
    return lo_vec
