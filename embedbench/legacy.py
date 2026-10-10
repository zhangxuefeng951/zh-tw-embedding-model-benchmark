"""匯入舊版 data.db（migrate.py 建立的 paragraphs / questions 表，向量存成 JSON 文字）。

對應方式：
- 段落：以段落文字比對新資料集的文件
- 問題：TCEval-v2 drcd 以 dataset_id 對應查詢編號；其他資料集以問題文字比對
匯入後用新的指標重新計算分數，並標示「舊資料匯入」。
"""

from __future__ import annotations

import ast
import json
import sqlite3
from pathlib import Path

import numpy as np

from embedbench.datasets import get_dataset
from embedbench.db import get_db
from embedbench.ranking import rank_dense
from embedbench.results import delete_result
from embedbench.runner import JobContext, save_result
from embedbench.vectors import vector_health


def _parse_vector(raw: str) -> np.ndarray:
    try:
        value = json.loads(raw)
    except (ValueError, TypeError):
        value = ast.literal_eval(raw)
    return np.asarray(value, dtype=np.float32).reshape(-1)


def list_legacy_models(path: str) -> list[dict]:
    conn = sqlite3.connect(f"file:{Path(path).expanduser()}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT q.model, COUNT(*), (SELECT COUNT(*) FROM paragraphs p WHERE p.model = q.model) "
            "FROM questions q GROUP BY q.model ORDER BY q.model"
        ).fetchall()
    finally:
        conn.close()
    return [{"model": m, "questions": nq, "paragraphs": npara} for m, nq, npara in rows]


def import_legacy(path: str, dataset_key: str, models: list[str] | None, ctx: JobContext) -> list[str]:
    db = get_db()
    ds = get_dataset(dataset_key, progress=lambda st, d, t: ctx.progress(f"準備資料集：{st}", d, t))
    conn = sqlite3.connect(f"file:{Path(path).expanduser()}?mode=ro", uri=True)
    result_ids = []
    try:
        available = [r["model"] for r in list_legacy_models(path)]
        targets = [m for m in (models or available) if m in available]
        if not targets:
            raise ValueError("舊資料庫中沒有可匯入的模型")
        doc_by_text: dict[str, int] = {}
        for i, d in enumerate(ds.documents):
            doc_by_text.setdefault(d.text.strip(), i)
        eval_pos = {q.id: i for i, q in enumerate(ds.eval_queries)}
        q_by_text: dict[str, list[int]] = {}
        for i, q in enumerate(ds.eval_queries):
            q_by_text.setdefault(q.text.strip(), []).append(i)
        use_dataset_id = dataset_key == "tceval-drcd-test"

        for mi, model in enumerate(targets):
            ctx.check_cancel()
            ctx.progress(f"匯入 {model}", mi, len(targets))
            # ---- 段落向量
            doc_vecs: np.ndarray | None = None
            filled = np.zeros(len(ds.documents), dtype=bool)
            unmatched_p = 0
            para_doc: dict[int, int] = {}
            for pid, content, emb in conn.execute(
                "SELECT id, content, embedding FROM paragraphs WHERE model = ? ORDER BY id", (model,)
            ):
                idx = doc_by_text.get((content or "").strip())
                if idx is None:
                    unmatched_p += 1
                    continue
                para_doc[pid] = idx
                if filled[idx]:
                    continue
                vec = _parse_vector(emb)
                if doc_vecs is None:
                    doc_vecs = np.zeros((len(ds.documents), len(vec)), dtype=np.float32)
                if len(vec) != doc_vecs.shape[1]:
                    raise ValueError(f"{model}：段落向量維度不一致")
                doc_vecs[idx] = vec
                filled[idx] = True
            if doc_vecs is None:
                ctx.log(f"{model}：沒有任何段落能對應到資料集 {dataset_key}，略過")
                continue
            missing_docs = int((~filled).sum())

            # ---- 問題向量
            q_vecs: dict[int, np.ndarray] = {}
            unmatched_q = 0
            for dataset_id, content, emb, paragraph_id in conn.execute(
                "SELECT dataset_id, content, embedding, paragraph_id FROM questions WHERE model = ? ORDER BY id",
                (model,),
            ):
                pos = eval_pos.get(str(dataset_id)) if use_dataset_id else None
                if pos is None:
                    candidates = q_by_text.get((content or "").strip(), [])
                    gold = para_doc.get(paragraph_id)
                    pos = next((c for c in candidates if gold in ds.relevant(ds.eval_queries[c].id)), None)
                    if pos is None and candidates:
                        pos = candidates[0]
                if pos is None:
                    unmatched_q += 1
                    continue
                vec = _parse_vector(emb)
                if len(vec) != doc_vecs.shape[1]:
                    raise ValueError(f"{model}：問題向量維度（{len(vec)}）與段落（{doc_vecs.shape[1]}）不一致")
                q_vecs.setdefault(pos, vec)
            if not q_vecs:
                ctx.log(f"{model}：沒有任何問題能對應到資料集 {dataset_key}，略過")
                continue
            positions = np.asarray(sorted(q_vecs), dtype=np.int64)
            qv = np.vstack([q_vecs[int(p)] for p in positions])
            relevant = [ds.relevant(ds.eval_queries[int(p)].id) for p in positions]
            ranking = rank_dense(qv, doc_vecs, relevant, check_cancel=ctx.check_cancel)
            if len(positions) != len(ds.eval_queries) or not np.array_equal(positions, np.arange(len(positions))):
                ranking.query_index = positions
            notes = []
            if missing_docs:
                notes.append(f"{missing_docs} 篇文件沒有舊向量（以零向量代替）")
            if unmatched_q:
                notes.append(f"{unmatched_q} 題對不上資料集，未納入")
            if len(positions) < len(ds.eval_queries):
                notes.append(f"只涵蓋 {len(positions)} / {len(ds.eval_queries)} 題")
            for n in notes:
                ctx.log(f"{model}：{n}")
            # 同一個舊模型重複匯入時，取代舊的匯入結果
            for old in db.query(
                "SELECT id FROM results WHERE kind = 'legacy' AND model = ? AND dataset_key = ?", (model, ds.key)
            ):
                delete_result(old["id"])
            result_ids.append(
                save_result(
                    db,
                    ranking,
                    {
                        "job_id": ctx.job_id,
                        "kind": "legacy",
                        "dataset_key": ds.key,
                        "label": f"{model}（舊資料匯入）",
                        "model": model,
                        "provider": "legacy",
                        "pair_label": "舊資料匯入（前綴未記錄）",
                        "query_prefix": None,
                        "document_prefix": None,
                        "dims": int(doc_vecs.shape[1]),
                        "num_documents": len(ds.documents),
                        "config": {
                            "source_db": str(path),
                            "legacy_model": model,
                            "unmatched_paragraphs": unmatched_p,
                            "unmatched_questions": unmatched_q,
                            "missing_documents": missing_docs,
                            "notes": notes,
                            "note": "舊資料匯入：前綴、端點、維度等設定未記錄",
                        },
                        "health": {"query": vector_health(qv), "document": vector_health(doc_vecs[filled])},
                    },
                )
            )
        ctx.progress("匯入完成", len(targets), len(targets))
    finally:
        conn.close()
    return result_ids
