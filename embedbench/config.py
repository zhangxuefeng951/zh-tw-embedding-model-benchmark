"""路徑與環境變數設定。"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent.parent


def data_dir() -> Path:
    """所有執行期資料（資料集、向量快取、結果、SQLite）都放在這裡，預設為專案下的 data/。"""
    path = Path(os.environ.get("EMBEDBENCH_DATA_DIR", ROOT / "data"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def presets_dir() -> Path:
    return Path(os.environ.get("EMBEDBENCH_PRESETS_DIR", ROOT / "presets"))


def env_file() -> Path:
    return Path(os.environ.get("EMBEDBENCH_ENV_FILE", ROOT / ".env"))


def get_env(name: str) -> str | None:
    """讀取環境變數；系統環境變數優先，其次是 .env 檔（每次重新讀取，修改 .env 不必重啟）。"""
    if not name:
        return None
    value = os.environ.get(name)
    if value:
        return value
    path = env_file()
    if path.exists():
        value = dotenv_values(path).get(name)
        if value:
            return value
    return None


def require_env(name: str) -> str:
    value = get_env(name)
    if not value:
        raise ValueError(f"環境變數 {name} 未設定，請在 .env 或系統環境變數中設定")
    return value


# 設定頁會顯示這些常用環境變數「是否已設定」（不顯示內容）
KNOWN_ENV_VARS = [
    "OPENAI_API_KEY",
    "COHERE_API_KEY",
    "VOYAGE_API_KEY",
    "GEMINI_API_KEY",
    "GCP_PROJECT_ID",
    "GCP_ACCESS_TOKEN",
    "MISTRAL_API_KEY",
    "JINA_API_KEY",
    "HF_TOKEN",
]
