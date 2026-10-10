import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import mock_server  # noqa: E402

TOPICS = [
    ("玉山", "玉山是臺灣最高峰，主峰海拔三千九百五十二公尺，位於南投縣與嘉義縣交界。", ["臺灣最高的山是哪一座？", "玉山主峰海拔多高？"]),
    ("日月潭", "日月潭位於南投縣魚池鄉，是臺灣最大的天然淡水湖泊，湖面以拉魯島為界分成日潭與月潭。", ["臺灣最大的天然淡水湖是哪一個？", "日月潭以哪個島分成日潭與月潭？"]),
    ("臺北101", "臺北101是位於臺北市信義區的摩天大樓，樓高五百零八公尺，曾是世界最高的建築物。", ["臺北101位於哪一區？", "臺北101樓高多少公尺？"]),
    ("阿里山", "阿里山森林鐵路從嘉義市出發，沿途經過熱帶、暖帶與溫帶林，以日出與雲海聞名。", ["阿里山森林鐵路從哪裡出發？", "阿里山以什麼景色聞名？"]),
    ("太魯閣", "太魯閣國家公園以立霧溪切割而成的大理石峽谷聞名，橫跨花蓮縣、臺中市與南投縣。", ["太魯閣峽谷是由哪條溪切割而成？", "太魯閣國家公園橫跨哪些縣市？"]),
    ("墾丁", "墾丁國家公園位於屏東縣恆春半島，是臺灣第一座國家公園，擁有珊瑚礁海岸。", ["臺灣第一座國家公園是哪一座？", "墾丁位於哪個半島？"]),
    ("故宮", "國立故宮博物院位於臺北市士林區，收藏翠玉白菜與肉形石等文物。", ["翠玉白菜收藏在哪裡？", "故宮博物院位於臺北市哪一區？"]),
    ("淡水", "淡水老街位於新北市淡水區，以紅毛城、漁人碼頭與夕陽景色著名。", ["紅毛城位於哪裡？", "淡水以什麼景色著名？"]),
]


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    d = tmp_path / "data"
    monkeypatch.setenv("EMBEDBENCH_DATA_DIR", str(d))
    monkeypatch.setenv("EMBEDBENCH_ENV_FILE", str(tmp_path / ".env"))
    import embedbench.datasets as datasets
    import embedbench.results as results

    datasets._cache.clear()
    results._rank_cache.clear()
    return d


@pytest.fixture
def mock_api():
    server, state, url = mock_server.start()
    yield state, url
    server.shutdown()


def tiny_rows():
    rows = []
    for title, paragraph, questions in TOPICS:
        for q in questions:
            rows.append({"question": q, "paragraph": paragraph, "title": title})
    return rows


@pytest.fixture
def tiny_dataset():
    import csv
    import io

    from embedbench.datasets import import_simple

    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=["question", "paragraph", "title"])
    w.writeheader()
    w.writerows(tiny_rows())
    return import_simple(buf.getvalue().encode("utf-8"), "tiny.csv", "tiny")
