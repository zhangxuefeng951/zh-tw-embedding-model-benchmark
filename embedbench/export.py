"""匯出排行榜與單一結果：CSV（Excel 可直接開啟）、JSON、Markdown 表格（方便貼部落格）。"""

from __future__ import annotations

import csv
import io
import json

from embedbench import prefixes
from embedbench.results import load_ranking, query_details, result_detail

METRICS = ("hit", "mrr", "ndcg", "recall")
METRIC_LABELS = {"hit": "Hit", "mrr": "MRR", "ndcg": "NDCG", "recall": "Recall"}


def metric_columns(ks: list[int], metrics: tuple[str, ...] = METRICS) -> list[str]:
    return [f"{m}@{k}" for m in metrics for k in ks]


def _row(r: dict, ks: list[int]) -> dict:
    row = {
        "標籤": r["label"],
        "類型": r["kind_label"],
        "模型": r.get("model") or "",
        "Provider": r.get("provider") or "",
        "前綴組合": r.get("pair_label") or "",
        "查詢前綴": "" if r.get("query_prefix") is None else prefixes.display(r["query_prefix"]),
        "文件前綴": "" if r.get("document_prefix") is None else prefixes.display(r["document_prefix"]),
        "維度": r.get("dims") or "",
        "題數": r["num_queries"],
        "資料集": r["dataset_key"],
        "結果 ID": r["id"],
        "建立時間": r["created_at"],
    }
    for col in metric_columns(ks):
        v = r["metrics"].get(col)
        row[col] = "" if v is None else round(v, 4)
    return row


def leaderboard_csv(rows: list[dict], ks: list[int]) -> str:
    buf = io.StringIO()
    data = [_row(r, ks) for r in rows]
    if data:
        w = csv.DictWriter(buf, fieldnames=list(data[0]))
        w.writeheader()
        w.writerows(data)
    return "\ufeff" + buf.getvalue()


def leaderboard_json(rows: list[dict], ks: list[int]) -> str:
    return json.dumps({"ks": ks, "results": rows}, ensure_ascii=False, indent=2)


def _md_escape(s) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ")


def leaderboard_markdown(rows: list[dict], ks: list[int], metrics: tuple[str, ...] = ("hit", "mrr")) -> str:
    cols = metric_columns(ks, metrics)
    header = ["#", "模型 / 標籤", "前綴組合", "維度", "題數"] + [
        f"{METRIC_LABELS[c.split('@')[0]]}@{c.split('@')[1]}" for c in cols
    ]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * 3 + ["---:"] * (len(header) - 3)) + "|"]
    for i, r in enumerate(rows, start=1):
        values = [
            str(i),
            _md_escape(r["label"]),
            _md_escape(r.get("pair_label") or ""),
            str(r.get("dims") or ""),
            str(r["num_queries"]),
        ]
        for c in cols:
            v = r["metrics"].get(c)
            values.append("" if v is None else f"{v:.4f}")
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def result_json(result_id: str, ks: list[int]) -> str:
    return json.dumps(result_detail(result_id, ks), ensure_ascii=False, indent=2)


def result_queries_csv(result_id: str, k: int = 5) -> str:
    """逐題結果：每題的正解名次、是否命中、前 5 名文件。"""
    r = load_ranking(result_id)
    detail = query_details(result_id, k=k, per_page=r.num_queries, top_n=5)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["題目 ID", "問題", "正解文件 ID", "正解最佳名次", f"命中@{k}", "前 5 名文件 ID", "前 5 名分數"])
    for item in detail["items"]:
        w.writerow(
            [
                item["query_id"],
                item["query"],
                " ".join(g["id"] for g in item["gold"]),
                item["best_rank"],
                int(item["hit"]),
                " ".join(t["id"] for t in item["top"]),
                " ".join("" if t["score"] is None else f"{t['score']:.4f}" for t in item["top"]),
            ]
        )
    return "\ufeff" + buf.getvalue()
