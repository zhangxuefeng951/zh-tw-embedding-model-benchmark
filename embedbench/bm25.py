"""BM25 基準（jieba 斷詞）。

計分公式與舊版使用的 rank_bm25.BM25Okapi 完全相同（含負 idf 以 epsilon × 平均 idf 取代），
但改用 scipy 稀疏矩陣計算，C-MTEB 這種 10 萬筆文件庫也跑得動。
"""

from __future__ import annotations

import json
import logging
import math
import warnings
from collections import Counter
from collections.abc import Callable

import numpy as np
from scipy import sparse

from embedbench import config
from embedbench.vectors import cache_key


def tokenize(texts: list[str], progress: Callable[[int, int], None] | None = None) -> list[list[str]]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)  # jieba 原始碼在 Python 3.12+ 的跳脫字元警告
        import jieba

    jieba.setLogLevel(logging.WARNING)
    out = []
    for i, t in enumerate(texts):
        out.append(jieba.lcut(t))
        if progress and (i + 1) % 500 == 0:
            progress(i + 1, len(texts))
    if progress:
        progress(len(texts), len(texts))
    return out


class BM25Index:
    def __init__(self, matrix: sparse.csr_matrix, vocab: dict[str, int]):
        self.matrix = matrix  # (num_docs, vocab) 已乘上 idf 與長度正規化的權重
        self.vocab = vocab

    @classmethod
    def build(
        cls,
        corpus_tokens: list[list[str]],
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ) -> BM25Index:
        vocab: dict[str, int] = {}
        rows, cols, tfs = [], [], []
        doc_len = np.zeros(len(corpus_tokens), dtype=np.float64)
        for d, tokens in enumerate(corpus_tokens):
            doc_len[d] = len(tokens)
            for tok, tf in Counter(tokens).items():
                j = vocab.setdefault(tok, len(vocab))
                rows.append(d)
                cols.append(j)
                tfs.append(tf)
        n_docs = len(corpus_tokens)
        rows_a = np.asarray(rows, dtype=np.int64)
        cols_a = np.asarray(cols, dtype=np.int64)
        tf_a = np.asarray(tfs, dtype=np.float64)
        df = np.bincount(cols_a, minlength=len(vocab)).astype(np.float64)
        idf = np.log(n_docs - df + 0.5) - np.log(df + 0.5)
        average_idf = idf.sum() / max(len(idf), 1)
        idf[idf < 0] = epsilon * average_idf
        avgdl = doc_len.sum() / max(n_docs, 1)
        norm = k1 * (1 - b + b * doc_len[rows_a] / avgdl)
        weights = idf[cols_a] * tf_a * (k1 + 1) / (tf_a + norm)
        matrix = sparse.csr_matrix((weights, (rows_a, cols_a)), shape=(n_docs, len(vocab)))
        return cls(matrix, vocab)

    def query_matrix(self, query_tokens: list[list[str]]) -> sparse.csr_matrix:
        rows, cols, vals = [], [], []
        for i, tokens in enumerate(query_tokens):
            for tok, c in Counter(tokens).items():
                j = self.vocab.get(tok)
                if j is not None:
                    rows.append(i)
                    cols.append(j)
                    vals.append(float(c))  # rank_bm25 會重複累加同一個查詢詞
        return sparse.csr_matrix((vals, (rows, cols)), shape=(len(query_tokens), len(self.vocab)))

    def scores(self, qmat: sparse.csr_matrix) -> np.ndarray:
        return np.asarray((qmat @ self.matrix.T).toarray(), dtype=np.float32)


def cached_index(
    dataset_key: str,
    documents_hash: str,
    doc_texts: list[str],
    k1: float,
    b: float,
    progress: Callable[[int, int], None] | None = None,
) -> BM25Index:
    key = cache_key({"dataset": dataset_key, "content": documents_hash, "k1": k1, "b": b, "tokenizer": "jieba"})
    directory = config.data_dir() / "bm25" / key
    mpath, vpath = directory / "matrix.npz", directory / "vocab.json"
    if mpath.exists() and vpath.exists():
        with open(vpath, encoding="utf-8") as f:
            return BM25Index(sparse.load_npz(mpath).tocsr(), json.load(f))
    index = BM25Index.build(tokenize(doc_texts, progress), k1=k1, b=b)
    directory.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(mpath, index.matrix)
    with open(vpath, "w", encoding="utf-8") as f:
        json.dump(index.vocab, f, ensure_ascii=False)
    return index


def reference_scores(corpus_tokens: list[list[str]], query: list[str], k1=1.5, b=0.75, epsilon=0.25) -> np.ndarray:
    """rank_bm25.BM25Okapi.get_scores 的直譯版本，只用於測試對照。"""
    n = len(corpus_tokens)
    doc_freqs = [Counter(d) for d in corpus_tokens]
    doc_len = np.array([len(d) for d in corpus_tokens], dtype=np.float64)
    avgdl = doc_len.sum() / n
    nd: Counter = Counter()
    for f in doc_freqs:
        nd.update(f.keys())
    idf = {w: math.log(n - c + 0.5) - math.log(c + 0.5) for w, c in nd.items()}
    eps = epsilon * (sum(idf.values()) / len(idf))
    idf = {w: (eps if v < 0 else v) for w, v in idf.items()}
    score = np.zeros(n)
    for q in query:
        q_freq = np.array([f.get(q, 0) for f in doc_freqs], dtype=np.float64)
        score += idf.get(q, 0) * (q_freq * (k1 + 1) / (q_freq + k1 * (1 - b + b * doc_len / avgdl)))
    return score
