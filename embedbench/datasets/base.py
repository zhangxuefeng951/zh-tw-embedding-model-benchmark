"""統一的檢索資料集格式（與 BEIR 相同的概念：文件庫、查詢、正解對應）。

磁碟格式（data/datasets/<key>/）：
- corpus.jsonl   每行 {"_id", "title", "text"}
- queries.jsonl  每行 {"_id", "text"}
- qrels.tsv      表頭 query-id / corpus-id / score
- meta.json      名稱、來源、說明、內容雜湊等
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path


@dataclass
class Document:
    id: str
    text: str
    title: str = ""


@dataclass
class Query:
    id: str
    text: str


@dataclass
class Dataset:
    key: str
    name: str
    documents: list[Document]
    queries: list[Query]
    qrels: dict[str, dict[str, int]]
    meta: dict = field(default_factory=dict)

    @cached_property
    def doc_index(self) -> dict[str, int]:
        return {d.id: i for i, d in enumerate(self.documents)}

    @cached_property
    def eval_queries(self) -> list[Query]:
        """有正解（score > 0 且正解文件存在於文件庫）的查詢才納入評測，順序與原始資料一致。"""
        return [
            q
            for q in self.queries
            if any(s > 0 and d in self.doc_index for d, s in self.qrels.get(q.id, {}).items())
        ]

    @property
    def has_titles(self) -> bool:
        return any(d.title for d in self.documents)

    def relevant(self, query_id: str) -> dict[int, float]:
        """回傳 {文件索引: 相關分數}，只包含 score > 0 且存在於文件庫的文件。"""
        out = {}
        for doc_id, score in self.qrels.get(query_id, {}).items():
            idx = self.doc_index.get(doc_id)
            if idx is not None and score > 0:
                out[idx] = float(score)
        return out

    @cached_property
    def documents_hash(self) -> str:
        h = hashlib.sha256()
        for d in self.documents:
            h.update(f"{d.id}\x1f{d.title}\x1f{d.text}\x1e".encode())
        return h.hexdigest()[:16]

    @cached_property
    def queries_hash(self) -> str:
        h = hashlib.sha256()
        for q in self.eval_queries:
            h.update(f"{q.id}\x1f{q.text}\x1e".encode())
        return h.hexdigest()[:16]

    def summary(self) -> dict:
        return {
            "key": self.key,
            "name": self.name,
            "num_documents": len(self.documents),
            "num_queries": len(self.eval_queries),
            "has_titles": self.has_titles,
            "documents_hash": self.documents_hash,
            "queries_hash": self.queries_hash,
        }


def save_dataset(ds: Dataset, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    tmp = directory / ".writing"
    tmp.touch()
    with open(directory / "corpus.jsonl", "w", encoding="utf-8") as f:
        for d in ds.documents:
            f.write(json.dumps({"_id": d.id, "title": d.title, "text": d.text}, ensure_ascii=False) + "\n")
    with open(directory / "queries.jsonl", "w", encoding="utf-8") as f:
        for q in ds.queries:
            f.write(json.dumps({"_id": q.id, "text": q.text}, ensure_ascii=False) + "\n")
    with open(directory / "qrels.tsv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["query-id", "corpus-id", "score"])
        for qid, rels in ds.qrels.items():
            for did, score in rels.items():
                w.writerow([qid, did, score])
    meta = dict(ds.meta)
    meta.update(ds.summary())
    with open(directory / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    tmp.unlink()


def is_saved(directory: Path) -> bool:
    return (directory / "meta.json").exists() and not (directory / ".writing").exists()


def read_meta(directory: Path) -> dict:
    with open(directory / "meta.json", encoding="utf-8") as f:
        return json.load(f)


def load_saved_dataset(directory: Path) -> Dataset:
    meta = read_meta(directory)
    documents = []
    with open(directory / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                o = json.loads(line)
                documents.append(Document(str(o["_id"]), o.get("text") or "", o.get("title") or ""))
    queries = []
    with open(directory / "queries.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                o = json.loads(line)
                queries.append(Query(str(o["_id"]), o.get("text") or ""))
    qrels = read_qrels_tsv(directory / "qrels.tsv")
    return Dataset(meta["key"], meta.get("name", meta["key"]), documents, queries, qrels, meta)


def read_qrels_tsv(path: Path) -> dict[str, dict[str, int]]:
    with open(path, encoding="utf-8") as f:
        return parse_qrels_lines(f)


def parse_qrels_lines(lines: Iterable[str]) -> dict[str, dict[str, int]]:
    """支援 BEIR（query-id corpus-id score，可有表頭）與 TREC（qid Q0 docid score）格式。"""
    qrels: dict[str, dict[str, int]] = {}
    for i, row in enumerate(csv.reader(lines, delimiter="\t")):
        if len(row) == 1:
            row = row[0].split()
        if not row or len(row) < 2:
            continue
        if i == 0 and not _is_number(row[-1]):
            continue  # 表頭
        if len(row) >= 4:
            qid, did, score = row[0], row[2], row[3]
        elif len(row) == 3:
            qid, did, score = row
        else:
            qid, did, score = row[0], row[1], 1
        qrels.setdefault(str(qid), {})[str(did)] = int(float(score))
    return qrels


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False
