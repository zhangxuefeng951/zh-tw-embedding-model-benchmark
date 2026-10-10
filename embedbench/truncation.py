"""輸入超過模型長度上限時的處理（保留前綴與開頭，從尾端截短）。

兩種方式，依序嘗試：
1. tokenizer 精確截斷（優先）：從 Hugging Face 下載模型的 tokenizer.json，送出前就把文字
   截到上限的 token 數，效果與 sentence-transformers 的預設截斷完全相同，也不會產生被拒絕的請求。
   上限可以在參數指定，或在服務端第一次回報「過長」時從錯誤訊息自動取得。
2. 試探法（拿不到 tokenizer 時）：一批文字被拒絕時對半拆開找出過長的文字，
   再以二分搜尋找出服務端能接受的最長開頭（誤差約 2%）。

注意：vLLM 對過長的輸入只回報「at least 上限+1 個 token」，這是下限而不是實際長度，
不能拿來推算要截掉多少；只有 TEI（Given: N）等回報實際長度的訊息才拿來當第一次的猜測。
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
TOLERANCE = 0.02  # 試探法找到的長度與最大可接受長度的差距上限（比例）


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


# ---- tokenizer 精確截斷 ------------------------------------------------------


class TokenTruncator:
    """用 tokenizers 的 Tokenizer 把文字截到 max_tokens（含 [CLS] / </s> 等特殊 token）。"""

    def __init__(self, tokenizer, name: str = ""):
        self.tokenizer = tokenizer
        self.name = name
        tokenizer.no_truncation()
        tokenizer.no_padding()

    def truncate(self, text: str, max_tokens: int) -> str:
        enc = self.tokenizer.encode(text)
        if len(enc.ids) <= max_tokens:
            return text
        special = sum(enc.special_tokens_mask)
        keep = max_tokens - special
        if keep <= 0:
            return text[:0]
        content = [o for o, sp in zip(enc.offsets, enc.special_tokens_mask) if not sp]
        return text[: content[keep - 1][1]]


def load_tokenizer(name: str) -> TokenTruncator:
    """從 Hugging Face 下載 tokenizer.json（遵循 HF_TOKEN、HF_ENDPOINT、HF_HOME 設定）。"""
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    path = hf_hub_download(name, "tokenizer.json")
    return TokenTruncator(Tokenizer.from_file(path), name)


class TruncationState:
    """同一個 provider 共用的狀態：長度上限、tokenizer、試探法學到的「每個 token 幾個字元」。"""

    def __init__(self, tokenizer_name: str | None = None, limit: int | None = None):
        self._lock = threading.Lock()
        self.tokenizer_name = (tokenizer_name or "").strip() or None
        self.limit = limit
        self._truncator: TokenTruncator | None = None
        self._tokenizer_tried = False
        self.notices: list[str] = []
        self._chars_per_token: list[float] = []

    # 長度上限
    def observe(self, e: ApiError) -> None:
        limit, _ = parse_limits(e)
        with self._lock:
            if limit and not self.limit:
                self.limit = limit
                self.notices.append(f"服務端回報輸入上限為 {limit} tokens")

    # tokenizer（只載入一次；失敗就改用試探法）
    def truncator(self) -> TokenTruncator | None:
        with self._lock:
            if self._tokenizer_tried or not self.tokenizer_name:
                return self._truncator
            self._tokenizer_tried = True
            try:
                self._truncator = load_tokenizer(self.tokenizer_name)
                self.notices.append(f"使用 tokenizer {self.tokenizer_name} 精確截斷過長的文字")
            except Exception as e:  # 離線、沒有權限、沒有 tokenizer.json…
                self.notices.append(
                    f"無法載入 tokenizer {self.tokenizer_name}（{e.__class__.__name__}: {str(e)[:120]}），"
                    "改用試探法截斷（較慢）"
                )
            return self._truncator

    def set_truncator(self, truncator: TokenTruncator | None) -> None:
        with self._lock:
            self._truncator, self._tokenizer_tried = truncator, True

    def prepare(self, texts: list[str]) -> list[str]:
        """上限已知且有 tokenizer 時，先把每筆文字截到上限。"""
        if not self.limit:
            return texts
        truncator = self.truncator()
        if truncator is None:
            return texts
        return [truncator.truncate(t, self.limit) for t in texts]

    def pop_notices(self) -> list[str]:
        with self._lock:
            out, self.notices = self.notices, []
        return out

    # 試探法
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
    """呼叫 request(texts)；過長的文字自動截短。on_truncate(原文, 實際送出的文字) 用來記錄。"""
    state = state or TruncationState()
    vecs, sent = _embed(state.prepare(texts), request, state)
    for original, final in zip(texts, sent):
        if final != original:
            on_truncate(original, final)
    return vecs


def _embed(texts: list[str], request, state: TruncationState) -> tuple[np.ndarray, list[str]]:
    try:
        return request(texts), texts
    except ApiError as e:
        if not is_context_error(e):
            raise
        error = e
    state.observe(error)
    prepared = state.prepare(texts)
    if prepared != texts:  # 剛得知上限、可以用 tokenizer 精確截斷
        try:
            return request(prepared), prepared
        except ApiError as e:
            if not is_context_error(e):
                raise
            texts, error = prepared, e
    if len(texts) > 1:
        mid = len(texts) // 2
        left, left_sent = _embed(texts[:mid], request, state)
        right, right_sent = _embed(texts[mid:], request, state)
        return np.vstack([left, right]), left_sent + right_sent
    vec, kept = _search_longest_prefix(texts[0], request, error, state)
    return vec, [kept]


def _initial_guess(length: int, error: ApiError, state: TruncationState) -> int:
    limit, actual = parse_limits(error)
    if limit and actual:
        return int(length * limit / actual * 0.98)
    cpt = state.estimate()
    if limit and cpt:
        return int(limit * cpt * 0.98)
    return length // 2


def _search_longest_prefix(
    original: str, request, error: ApiError, state: TruncationState
) -> tuple[np.ndarray, str]:
    """試探法：二分搜尋最長可接受的開頭。lo＝已知可接受的長度、hi＝已知會被拒絕的長度。"""
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
    return lo_vec, original[:lo]
