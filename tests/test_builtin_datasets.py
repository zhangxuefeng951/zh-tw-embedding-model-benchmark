"""內建資料集轉換邏輯（以假的 Hugging Face load_dataset 測試，不需要網路）。"""

import sys
import types

import pytest

from embedbench.datasets import get_dataset, list_datasets, prepare_dataset


@pytest.fixture
def fake_hf(monkeypatch):
    tables = {}

    def load_dataset(name, config=None, split=None):
        key = (name, config, split)
        if key not in tables:
            raise ValueError(f"no such split {key}")
        return tables[key]

    module = types.ModuleType("datasets")
    module.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", module)
    return tables


def test_tceval_drcd_dedupes_paragraphs(fake_hf):
    fake_hf[("MediaTek-Research/TCEval-v2", "drcd", "test")] = [
        {"paragraph": "段落一", "question": "問一"},
        {"paragraph": "段落一", "question": "問二"},
        {"paragraph": "段落二", "question": "問三"},
        {"paragraph": "段落一", "question": "問四"},  # 不連續出現的相同段落也合併
    ]
    ds = prepare_dataset("tceval-drcd-test")
    assert len(ds.documents) == 2
    assert [q.id for q in ds.eval_queries] == ["0", "1", "2", "3"]  # 與舊 data.db 的 dataset_id 一致
    assert ds.relevant("3") == ds.relevant("0")
    assert not ds.has_titles


def test_cmteb_and_opencc_variant(fake_hf):
    fake_hf[("C-MTEB/CovidRetrieval", None, "corpus")] = [{"id": 1, "text": "新冠病毒疫苗软件"}, {"id": 2, "text": "其他内容"}]
    fake_hf[("C-MTEB/CovidRetrieval", None, "queries")] = [{"id": "q1", "text": "疫苗软件是什么"}, {"id": "q2", "text": "没有正解"}]
    fake_hf[("C-MTEB/CovidRetrieval-qrels", None, "dev")] = [{"qid": "q1", "pid": 1, "score": 1}]
    ds = get_dataset("cmteb-covidretrieval-tw")
    assert ds.documents[0].text == "新冠病毒疫苗軟體"  # s2twp：簡轉繁 + 台灣用詞
    assert ds.queries[0].text.startswith("疫苗軟體是什麼")
    assert [q.id for q in ds.eval_queries] == ["q1"]
    assert ds.relevant("q1") == {0: 1.0}
    keys = {d["key"]: d for d in list_datasets()}
    assert keys["cmteb-covidretrieval"]["prepared"]  # 原文版本也一併準備好
    assert keys["cmteb-covidretrieval-tw"]["num_queries"] == 1


def test_builtin_list_contains_all_cmteb_subsets():
    keys = {d["key"] for d in list_datasets()}
    for subset in ("t2", "mmarco", "du", "covid", "cmedqa", "medical", "ecom", "video"):
        assert f"cmteb-{subset}retrieval" in keys
        assert f"cmteb-{subset}retrieval-tw" in keys
    assert {"drcd-test", "drcd-dev", "tceval-drcd-test"} <= keys
