"""資料集 registry：內建資料集（首次使用時下載並轉換）與自訂上傳資料集。"""

from __future__ import annotations

import shutil
import threading
from collections import OrderedDict
from pathlib import Path

from embedbench import config
from embedbench.datasets.base import Dataset, is_saved, load_saved_dataset, read_meta, save_dataset
from embedbench.datasets.builtin import ProgressFn, builtin_specs
from embedbench.datasets.custom import (
    make_custom_key,
    parse_beir,
    parse_beir_zip,
    parse_simple_file,
)

__all__ = [
    "Dataset",
    "dataset_dir",
    "delete_dataset",
    "get_dataset",
    "import_beir",
    "import_simple",
    "list_datasets",
    "prepare_dataset",
]

_cache: OrderedDict[str, tuple[float, Dataset]] = OrderedDict()
_cache_lock = threading.Lock()
_prepare_locks: dict[str, threading.Lock] = {}
CACHE_SIZE = 4


def datasets_root() -> Path:
    path = config.data_dir() / "datasets"
    path.mkdir(parents=True, exist_ok=True)
    return path


def dataset_dir(key: str) -> Path:
    if "/" in key or "\\" in key or key.startswith("."):
        raise ValueError(f"不合法的資料集名稱：{key}")
    return datasets_root() / key


def _custom_keys() -> list[str]:
    return sorted(p.name for p in datasets_root().iterdir() if p.is_dir() and p.name.startswith("custom-"))


def list_datasets() -> list[dict]:
    out = []
    for spec in builtin_specs().values():
        directory = dataset_dir(spec.key)
        item = {
            "key": spec.key,
            "name": spec.name,
            "group": spec.group,
            "description": spec.description,
            "source": spec.source,
            "license": spec.license,
            "size_hint": spec.size_hint,
            "builtin": True,
            "prepared": is_saved(directory),
        }
        if item["prepared"]:
            item.update(_counts(read_meta(directory)))
        out.append(item)
    for key in _custom_keys():
        directory = dataset_dir(key)
        if not is_saved(directory):
            continue
        meta = read_meta(directory)
        item = {
            "key": key,
            "name": meta.get("name", key),
            "group": "自訂",
            "description": meta.get("description", ""),
            "source": meta.get("filename", "上傳檔案"),
            "license": "",
            "size_hint": "",
            "builtin": False,
            "prepared": True,
        }
        item.update(_counts(meta))
        out.append(item)
    return out


def _counts(meta: dict) -> dict:
    return {
        "num_documents": meta.get("num_documents"),
        "num_queries": meta.get("num_queries"),
        "has_titles": meta.get("has_titles"),
    }


def dataset_exists(key: str) -> bool:
    return key in builtin_specs() or is_saved(dataset_dir(key))


def get_dataset(key: str, prepare: bool = True, progress: ProgressFn | None = None) -> Dataset:
    directory = dataset_dir(key)
    if not is_saved(directory):
        if not prepare:
            raise FileNotFoundError(f"資料集 {key} 尚未下載")
        return prepare_dataset(key, progress)
    mtime = (directory / "meta.json").stat().st_mtime
    with _cache_lock:
        hit = _cache.get(key)
        if hit and hit[0] == mtime:
            _cache.move_to_end(key)
            return hit[1]
    ds = load_saved_dataset(directory)
    _remember(key, mtime, ds)
    return ds


def _remember(key: str, mtime: float, ds: Dataset) -> None:
    with _cache_lock:
        _cache[key] = (mtime, ds)
        _cache.move_to_end(key)
        while len(_cache) > CACHE_SIZE:
            _cache.popitem(last=False)


def prepare_dataset(key: str, progress: ProgressFn | None = None, force: bool = False) -> Dataset:
    specs = builtin_specs()
    if key not in specs:
        if is_saved(dataset_dir(key)):
            return get_dataset(key)
        raise KeyError(f"找不到資料集：{key}")
    lock = _prepare_locks.setdefault(key, threading.Lock())
    with lock:
        directory = dataset_dir(key)
        if is_saved(directory) and not force:
            return get_dataset(key)
        spec = specs[key]
        ds = spec.builder(progress or (lambda *a: None))
        ds.key = key
        ds.name = spec.name
        ds.meta.update({"description": spec.description, "source": spec.source, "license": spec.license})
        save_dataset(ds, directory)
        _remember(key, (directory / "meta.json").stat().st_mtime, ds)
        return ds


def _save_custom(ds: Dataset, description: str, filename: str) -> Dataset:
    ds.meta.update({"description": description, "filename": filename, "custom": True})
    directory = dataset_dir(ds.key)
    save_dataset(ds, directory)
    return ds


def import_simple(content: bytes, filename: str, name: str, description: str = "") -> Dataset:
    key = make_custom_key(name, set(_custom_keys()))
    ds = parse_simple_file(content, filename, key, name)
    return _save_custom(ds, description, filename)


def import_beir(
    name: str,
    description: str = "",
    corpus: bytes | None = None,
    queries: bytes | None = None,
    qrels: bytes | None = None,
    zip_content: bytes | None = None,
) -> Dataset:
    key = make_custom_key(name, set(_custom_keys()))
    if zip_content is not None:
        ds = parse_beir_zip(zip_content, key, name)
        filename = "BEIR zip"
    else:
        if not (corpus and queries and qrels):
            raise ValueError("BEIR 格式需要 corpus.jsonl、queries.jsonl、qrels.tsv 三個檔案")
        ds = parse_beir(corpus, queries, qrels, key, name)
        filename = "BEIR"
    return _save_custom(ds, description, filename)


def delete_dataset(key: str) -> None:
    """自訂資料集：刪除；內建資料集：清除已下載的快取（下次使用時重新下載）。"""
    directory = dataset_dir(key)
    if directory.exists():
        shutil.rmtree(directory)
    with _cache_lock:
        _cache.pop(key, None)
