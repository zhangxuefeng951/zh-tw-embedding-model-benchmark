"""自訂資料集匯入：簡單格式（CSV / JSONL）與 BEIR 格式（corpus.jsonl + queries.jsonl + qrels.tsv，或 zip）。"""

from __future__ import annotations

import csv
import io
import json
import re
import zipfile
from pathlib import Path

from embedbench.datasets.base import Dataset, Document, Query, parse_qrels_lines

QUESTION_KEYS = ("question", "query", "問題")
PARAGRAPH_KEYS = ("paragraph", "document", "passage", "context", "text", "段落")
TITLE_KEYS = ("title", "標題")


def _pick(row: dict, keys: tuple[str, ...]) -> str | None:
    lowered = {str(k).strip().lower(): v for k, v in row.items() if k is not None}
    for k in keys:
        if k in lowered and lowered[k] is not None:
            return str(lowered[k])
    return None


def parse_simple_rows(rows: list[dict], key: str, name: str) -> Dataset:
    """每列一題：question、paragraph（必填），title（選填）。相同段落自動合併成同一篇文件。"""
    documents: list[Document] = []
    by_content: dict[tuple[str, str], str] = {}
    queries, qrels = [], {}
    for i, row in enumerate(rows, start=1):
        question = _pick(row, QUESTION_KEYS)
        paragraph = _pick(row, PARAGRAPH_KEYS)
        title = (_pick(row, TITLE_KEYS) or "").strip()
        if not question or not paragraph or not question.strip() or not paragraph.strip():
            raise ValueError(f"第 {i} 筆缺少 question 或 paragraph 欄位")
        content_key = (title, paragraph)
        doc_id = by_content.get(content_key)
        if doc_id is None:
            doc_id = f"d{len(documents)}"
            by_content[content_key] = doc_id
            documents.append(Document(doc_id, paragraph, title))
        qid = f"q{len(queries)}"
        queries.append(Query(qid, question.strip()))
        qrels[qid] = {doc_id: 1}
    if not queries:
        raise ValueError("檔案沒有任何資料列")
    return Dataset(key, name, documents, queries, qrels, {"format": "simple"})


def parse_simple_file(content: bytes, filename: str, key: str, name: str) -> Dataset:
    text = content.decode("utf-8-sig")
    suffix = Path(filename).suffix.lower()
    if suffix in (".jsonl", ".json"):
        stripped = text.strip()
        if stripped.startswith("["):
            rows = json.loads(stripped)
        else:
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    elif suffix in (".csv", ".tsv"):
        delimiter = "\t" if suffix == ".tsv" else ","
        rows = list(csv.DictReader(io.StringIO(text), delimiter=delimiter))
    else:
        raise ValueError("簡單格式只支援 .csv、.tsv、.jsonl、.json")
    return parse_simple_rows(rows, key, name)


def _read_jsonl(content: bytes) -> list[dict]:
    return [json.loads(line) for line in content.decode("utf-8-sig").splitlines() if line.strip()]


def parse_beir(corpus: bytes, queries: bytes, qrels: bytes, key: str, name: str) -> Dataset:
    documents = []
    for o in _read_jsonl(corpus):
        doc_id = o.get("_id", o.get("id"))
        if doc_id is None:
            raise ValueError("corpus.jsonl 每行都需要 _id 欄位")
        documents.append(Document(str(doc_id), o.get("text") or "", o.get("title") or ""))
    query_list = []
    for o in _read_jsonl(queries):
        qid = o.get("_id", o.get("id"))
        if qid is None:
            raise ValueError("queries.jsonl 每行都需要 _id 欄位")
        query_list.append(Query(str(qid), o.get("text") or ""))
    qrels_map = parse_qrels_lines(qrels.decode("utf-8-sig").splitlines())
    ds = Dataset(key, name, documents, query_list, qrels_map, {"format": "beir"})
    if not ds.eval_queries:
        raise ValueError("qrels 與 queries 對不上：沒有任何查詢有正解")
    return ds


def parse_beir_zip(content: bytes, key: str, name: str) -> Dataset:
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]

        def find(predicate) -> bytes:
            matches = [n for n in names if predicate(n.lower())]
            if not matches:
                raise ValueError("zip 內找不到必要檔案（corpus.jsonl、queries.jsonl、qrels/*.tsv）")
            return zf.read(sorted(matches, key=len)[0])

        corpus = find(lambda n: n.endswith("corpus.jsonl"))
        queries = find(lambda n: n.endswith("queries.jsonl"))
        # BEIR 標準結構為 qrels/test.tsv；優先 test，其次任何 tsv
        try:
            qrels = find(lambda n: n.endswith("qrels/test.tsv") or n.endswith("qrels.tsv"))
        except ValueError:
            qrels = find(lambda n: n.endswith(".tsv"))
    return parse_beir(corpus, queries, qrels, key, name)


def make_custom_key(name: str, existing: set[str]) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40]
    if not slug:
        import hashlib

        slug = hashlib.sha1(name.encode()).hexdigest()[:8]
    key = f"custom-{slug}"
    n = 2
    while key in existing:
        key = f"custom-{slug}-{n}"
        n += 1
    return key
