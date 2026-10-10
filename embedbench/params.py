"""Provider 參數定義：同一份定義同時驅動 Web UI 表單、CLI 設定檔驗證與快取鍵計算。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class Param:
    name: str
    label: str
    kind: str = "text"  # text | int | float | bool | select | json
    default: Any = None
    help: str = ""
    options: list | None = None
    affects_output: bool = True  # 會影響向量 / 分數的參數才納入快取鍵
    placeholder: str = ""
    advanced: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def coerce(spec: Param, value: Any) -> Any:
    if value is None or (isinstance(value, str) and value.strip() == "" and spec.kind != "text"):
        return spec.default
    try:
        match spec.kind:
            case "int":
                return int(value)
            case "float":
                return float(value)
            case "bool":
                if isinstance(value, str):
                    return value.strip().lower() in ("1", "true", "yes", "on")
                return bool(value)
            case "json":
                if isinstance(value, str):
                    value = json.loads(value) if value.strip() else spec.default
                if value is not None and not isinstance(value, dict):
                    raise ValueError("必須是 JSON 物件，例如 {\"key\": \"value\"}")
                return value
            case "select":
                if spec.options and value not in spec.options:
                    raise ValueError(f"必須是 {spec.options} 其中之一")
                return value
            case _:
                return str(value)
    except (ValueError, TypeError) as e:
        raise ValueError(f"參數「{spec.label}」（{spec.name}）格式錯誤：{e}") from e


def coerce_params(specs: list[Param], raw: dict | None) -> dict:
    raw = raw or {}
    out = {}
    for spec in specs:
        out[spec.name] = coerce(spec, raw.get(spec.name))
    return out


def output_signature(specs: list[Param], values: dict) -> dict:
    return {s.name: values.get(s.name) for s in specs if s.affects_output}


# ---- 常用參數 -------------------------------------------------------------

def model_param(default: str = "", help: str = "") -> Param:
    return Param("model", "模型名稱", "text", default, help or "送給 API 的 model 欄位，或本地模型的 Hugging Face 名稱 / 路徑")


def api_key_param(default: str = "") -> Param:
    return Param(
        "api_key_env",
        "API Key 環境變數名稱",
        "text",
        default,
        "填環境變數「名稱」（例如 OPENAI_API_KEY），Key 本身放在 .env 或系統環境變數。本地服務不需要 Key 可留空。",
        affects_output=False,
        placeholder="OPENAI_API_KEY",
    )


def batch_size_param(default: int = 32) -> Param:
    return Param("batch_size", "批次大小", "int", default, "每次請求送出的文字筆數", affects_output=False)


def concurrency_param(default: int = 4) -> Param:
    return Param("concurrency", "並行請求數", "int", default, "同時送出的請求數；遇到速率限制請調低", affects_output=False)


def timeout_param(default: float = 120.0) -> Param:
    return Param("timeout", "逾時（秒）", "float", default, affects_output=False, advanced=True)


def retries_param(default: int = 5) -> Param:
    return Param(
        "max_retries", "最大重試次數", "int", default, "遇到 429 / 5xx / 連線錯誤時以指數退避重試", affects_output=False, advanced=True
    )


def dimensions_param(field_hint: str = "dimensions") -> Param:
    return Param(
        "dimensions",
        "API 輸出維度",
        "int",
        None,
        f"送給 API 的 {field_hint} 參數（Matryoshka 降維）；留空＝模型預設維度。"
        "也可以在下方「評測維度」由系統截取，不必重新呼叫 API。",
        placeholder="留空＝預設",
    )


def extra_params() -> list[Param]:
    return [
        Param(
            "query_extra",
            "查詢額外參數（JSON）",
            "json",
            {},
            "合併進「查詢」請求的額外欄位，例如 Jina API 的 {\"task\": \"retrieval.query\"}",
            advanced=True,
        ),
        Param(
            "document_extra",
            "文件額外參數（JSON）",
            "json",
            {},
            "合併進「文件」請求的額外欄位，例如 {\"task\": \"retrieval.passage\"}",
            advanced=True,
        ),
    ]
