"""模型預設檔：專案內 presets/*.yaml（內建，跟著 git 版控）＋ 使用者在 Web UI 另存的自訂預設（本機資料庫）。

Embedding 預設檔欄位：
  id, name, description, provider, params, provider_params（切換 provider 時套用）,
  prefix_pairs（label / query / document / default）, suggested_dims, notes
Reranker 預設檔欄位：
  id, name, description, reranker, params, provider_params, candidates, notes
"""

from __future__ import annotations

import secrets

import yaml

from embedbench import config
from embedbench.db import get_db, now_iso

FILES = {"embedding": "embedding.yaml", "reranker": "reranker.yaml"}


def load_builtin(kind: str) -> list[dict]:
    path = config.presets_dir() / FILES[kind]
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    out = []
    for item in data.get("presets") or []:
        if not item.get("id") or not item.get("name"):
            raise ValueError(f"{path.name} 中有預設檔缺少 id 或 name：{item}")
        out.append(_normalize(kind, {**item, "source": "builtin"}))
    ids = [p["id"] for p in out]
    dup = {i for i in ids if ids.count(i) > 1}
    if dup:
        raise ValueError(f"{path.name} 中有重複的預設檔 id：{dup}")
    return out


def _normalize(kind: str, item: dict) -> dict:
    item = dict(item)
    item["kind"] = kind
    item.setdefault("description", "")
    item.setdefault("params", {})
    item.setdefault("provider_params", {})
    item.setdefault("notes", [])
    if isinstance(item["notes"], str):
        item["notes"] = [item["notes"]]
    if kind == "embedding":
        item.setdefault("provider", "openai")
        item.setdefault("suggested_dims", [])
        pairs = []
        for p in item.get("prefix_pairs") or []:
            pairs.append(
                {
                    "label": p.get("label") or "",
                    "query": p.get("query") or "",
                    "document": p.get("document") or "",
                    "default": bool(p.get("default", False)),
                }
            )
        if not pairs:
            pairs = [{"label": "無前綴", "query": "", "document": "", "default": True}]
        if not any(p["default"] for p in pairs):
            pairs[0]["default"] = True
        item["prefix_pairs"] = pairs
    else:
        item.setdefault("reranker", "rerank_api")
        item.setdefault("candidates", 50)
    return item


def list_presets(kind: str) -> list[dict]:
    builtin = load_builtin(kind)
    user = [
        _normalize(kind, {**row["data"], "id": row["id"], "source": "user", "updated_at": row["updated_at"]})
        for row in get_db().query(
            "SELECT * FROM user_presets WHERE kind = ? ORDER BY updated_at DESC", (kind,), table="user_presets"
        )
    ]
    return user + builtin


def get_preset(kind: str, preset_id: str) -> dict | None:
    return next((p for p in list_presets(kind) if p["id"] == preset_id), None)


def save_user_preset(kind: str, data: dict, preset_id: str | None = None) -> dict:
    if kind not in FILES:
        raise ValueError(f"未知的預設檔類型：{kind}")
    if not (data.get("name") or "").strip():
        raise ValueError("請填寫預設檔名稱")
    clean = {k: v for k, v in data.items() if k not in ("id", "source", "kind", "updated_at")}
    _normalize(kind, clean)  # 驗證格式
    db = get_db()
    now = now_iso()
    if preset_id and preset_id.startswith("user-") and db.query_one("SELECT id FROM user_presets WHERE id = ?", (preset_id,)):
        db.update("user_presets", preset_id, {"data": clean, "updated_at": now})
    else:
        preset_id = f"user-{secrets.token_hex(4)}"
        db.insert("user_presets", {"id": preset_id, "kind": kind, "data": clean, "created_at": now, "updated_at": now})
    return get_preset(kind, preset_id)


def delete_user_preset(preset_id: str) -> None:
    if not preset_id.startswith("user-"):
        raise ValueError("內建預設檔請直接編輯 presets/*.yaml")
    get_db().execute("DELETE FROM user_presets WHERE id = ?", (preset_id,))


def export_yaml(preset: dict) -> str:
    """把（自訂）預設檔轉成 YAML，方便貼進 presets/*.yaml 變成內建預設。"""
    data = {k: v for k, v in preset.items() if k not in ("source", "kind", "updated_at")}
    if str(data.get("id", "")).startswith("user-"):
        data["id"] = data["id"].removeprefix("user-")
    return yaml.safe_dump({"presets": [data]}, allow_unicode=True, sort_keys=False)
