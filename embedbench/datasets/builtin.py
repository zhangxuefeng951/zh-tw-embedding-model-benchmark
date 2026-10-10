"""內建資料集：DRCD（GitHub 原始版 test / dev）、TCEval-v2 drcd、C-MTEB 8 個檢索子集（簡體 / OpenCC 轉繁體）。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import requests

from embedbench.datasets.base import Dataset, Document, Query

ProgressFn = Callable[[str, int, int], None]


def _noop(stage: str, done: int, total: int) -> None:
    pass


@dataclass
class BuiltinSpec:
    key: str
    name: str
    group: str
    description: str
    source: str
    license: str
    builder: Callable[[ProgressFn], Dataset]
    size_hint: str = ""


# ---- DRCD（台達閱讀理解資料集，GitHub 原始版） -------------------------------

DRCD_URL = "https://raw.githubusercontent.com/DRCKnowledgeTeam/DRCD/master/DRCD_{split}.json"


def build_drcd(split: str, key: str, name: str) -> Callable[[ProgressFn], Dataset]:
    def builder(progress: ProgressFn = _noop) -> Dataset:
        progress(f"下載 DRCD_{split}.json", 0, 1)
        resp = requests.get(DRCD_URL.format(split=split), timeout=120)
        resp.raise_for_status()
        raw = resp.json()
        documents, queries, qrels = [], [], {}
        for article in raw["data"]:
            for para in article["paragraphs"]:
                documents.append(Document(para["id"], para["context"], article["title"]))
                for qa in para["qas"]:
                    queries.append(Query(qa["id"], qa["question"]))
                    qrels[qa["id"]] = {para["id"]: 1}
        progress(f"下載 DRCD_{split}.json", 1, 1)
        return Dataset(key, name, documents, queries, qrels, {"version": raw.get("version")})

    return builder


# ---- TCEval-v2 drcd（MediaTek Research，Hugging Face；舊版腳本使用的來源） -----


def build_tceval_drcd(progress: ProgressFn = _noop) -> Dataset:
    from datasets import load_dataset

    progress("從 Hugging Face 下載 TCEval-v2 drcd", 0, 1)
    rows = load_dataset("MediaTek-Research/TCEval-v2", "drcd", split="test")
    documents: list[Document] = []
    by_text: dict[str, str] = {}
    queries, qrels = [], {}
    for i, row in enumerate(rows):
        paragraph = row["paragraph"]
        doc_id = by_text.get(paragraph)
        if doc_id is None:
            doc_id = f"p{len(documents)}"
            by_text[paragraph] = doc_id
            documents.append(Document(doc_id, paragraph, ""))
        # 查詢編號沿用資料集索引，與舊版 data.db 的 dataset_id 一致
        qid = str(i)
        queries.append(Query(qid, row["question"]))
        qrels[qid] = {doc_id: 1}
    progress("從 Hugging Face 下載 TCEval-v2 drcd", 1, 1)
    return Dataset("tceval-drcd-test", "TCEval-v2 drcd（test）", documents, queries, qrels)


# ---- C-MTEB 檢索資料集 --------------------------------------------------------

CMTEB_SUBSETS = {
    "T2Retrieval": ("T2Ranking 網頁搜尋段落檢索", "問題→段落"),
    "MMarcoRetrieval": ("mMARCO（MS MARCO 中文版）", "問題→段落"),
    "DuRetrieval": ("百度 DuReader 網頁搜尋段落檢索", "問題→段落"),
    "CovidRetrieval": ("COVID-19 新聞", "問題→段落"),
    "CmedqaRetrieval": ("中文線上醫療問答", "問題→段落"),
    "MedicalRetrieval": ("醫療領域", "問題→段落"),
    "EcomRetrieval": ("電商商品標題", "關鍵字配對"),
    "VideoRetrieval": ("影片標題", "關鍵字配對"),
}


def _hf_split(name: str, split: str):
    from datasets import load_dataset

    try:
        return load_dataset(name, split=split)
    except Exception:
        # 部分鏡像使用 BEIR 風格的 config 名稱
        return load_dataset(name, split, split=split)


def build_cmteb(subset: str) -> Callable[[ProgressFn], Dataset]:
    def builder(progress: ProgressFn = _noop) -> Dataset:
        progress(f"從 Hugging Face 下載 C-MTEB/{subset}", 0, 3)
        corpus = _hf_split(f"C-MTEB/{subset}", "corpus")
        progress(f"從 Hugging Face 下載 C-MTEB/{subset}", 1, 3)
        queries_rows = _hf_split(f"C-MTEB/{subset}", "queries")
        progress(f"從 Hugging Face 下載 C-MTEB/{subset}", 2, 3)
        qrels_rows = _hf_split(f"C-MTEB/{subset}-qrels", "dev")
        documents = [Document(str(r["id"]), r["text"] or "", r.get("title") or "") for r in corpus]
        queries = [Query(str(r["id"]), r["text"] or "") for r in queries_rows]
        qrels: dict[str, dict[str, int]] = {}
        for r in qrels_rows:
            qrels.setdefault(str(r["qid"]), {})[str(r["pid"])] = int(r["score"])
        progress(f"從 Hugging Face 下載 C-MTEB/{subset}", 3, 3)
        return Dataset(f"cmteb-{subset.lower()}", f"C-MTEB {subset}（簡體原文）", documents, queries, qrels)

    return builder


def opencc_converter(config_name: str = "s2twp"):
    import opencc

    try:
        return opencc.OpenCC(config_name)
    except Exception:
        return opencc.OpenCC(f"{config_name}.json")


def build_cmteb_tw(subset: str) -> Callable[[ProgressFn], Dataset]:
    def builder(progress: ProgressFn = _noop) -> Dataset:
        from embedbench.datasets import get_dataset

        source = get_dataset(f"cmteb-{subset.lower()}", progress=progress)
        cc = opencc_converter("s2twp")
        total = len(source.documents) + len(source.queries)
        documents, done = [], 0
        for d in source.documents:
            documents.append(Document(d.id, cc.convert(d.text), cc.convert(d.title) if d.title else ""))
            done += 1
            if done % 2000 == 0:
                progress("OpenCC 簡轉繁（s2twp）", done, total)
        queries = [Query(q.id, cc.convert(q.text)) for q in source.queries]
        progress("OpenCC 簡轉繁（s2twp）", total, total)
        return Dataset(
            f"cmteb-{subset.lower()}-tw",
            f"C-MTEB {subset}（OpenCC 轉繁體）",
            documents,
            queries,
            source.qrels,
            {"converted_from": source.key, "opencc": "s2twp"},
        )

    return builder


def builtin_specs() -> dict[str, BuiltinSpec]:
    specs = [
        BuiltinSpec(
            "drcd-test",
            "DRCD test（GitHub 原始版）",
            "DRCD",
            "台達閱讀理解資料集測試集：3493 題、1000 段落，附文章標題。題目與 TCEval-v2 drcd 相同。",
            "https://github.com/DRCKnowledgeTeam/DRCD",
            "CC BY-SA 3.0",
            build_drcd("test", "drcd-test", "DRCD test（GitHub 原始版）"),
            "約 2 MB",
        ),
        BuiltinSpec(
            "drcd-dev",
            "DRCD dev（GitHub 原始版）",
            "DRCD",
            "台達閱讀理解資料集開發集：3524 題、1000 段落，附文章標題。",
            "https://github.com/DRCKnowledgeTeam/DRCD",
            "CC BY-SA 3.0",
            build_drcd("dev", "drcd-dev", "DRCD dev（GitHub 原始版）"),
            "約 2 MB",
        ),
        BuiltinSpec(
            "tceval-drcd-test",
            "TCEval-v2 drcd（test）",
            "DRCD",
            "舊版腳本使用的來源（MediaTek-Research/TCEval-v2），沒有標題欄位。匯入舊 data.db 時預設對應這個資料集。",
            "https://huggingface.co/datasets/MediaTek-Research/TCEval-v2",
            "依 Hugging Face 資料集頁面",
            build_tceval_drcd,
            "約 2 MB",
        ),
    ]
    for subset, (desc, kind) in CMTEB_SUBSETS.items():
        specs.append(
            BuiltinSpec(
                f"cmteb-{subset.lower()}",
                f"C-MTEB {subset}（簡體原文）",
                "C-MTEB",
                f"{desc}；{kind}。簡體中文原文，可與 C-MTEB 官方分數比較。文件庫約 10 萬筆，評測成本較高。",
                f"https://huggingface.co/datasets/C-MTEB/{subset}",
                "依 Hugging Face 資料集頁面",
                build_cmteb(subset),
                "文件庫約 10 萬筆",
            )
        )
        specs.append(
            BuiltinSpec(
                f"cmteb-{subset.lower()}-tw",
                f"C-MTEB {subset}（OpenCC 轉繁體）",
                "C-MTEB",
                f"{desc}；{kind}。以 OpenCC s2twp（含台灣常用詞）將文件與查詢轉成繁體。",
                f"https://huggingface.co/datasets/C-MTEB/{subset}",
                "依 Hugging Face 資料集頁面",
                build_cmteb_tw(subset),
                "文件庫約 10 萬筆",
            )
        )
    return {s.key: s for s in specs}
