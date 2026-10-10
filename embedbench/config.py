"""路徑與環境變數設定。

設定來源的優先順序：系統環境變數 > 專案根目錄的 .env 檔 > 預設值。
.env 每次讀取時都會檢查是否有修改，改 API Key 不必重啟。範本見 .env.example。
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent.parent

_dotenv_lock = threading.Lock()
_dotenv_cache: tuple[Path, float, dict] | None = None


def env_file() -> Path:
    return Path(os.environ.get("EMBEDBENCH_ENV_FILE", ROOT / ".env"))


def _dotenv() -> dict:
    """讀取 .env（依修改時間快取）。"""
    global _dotenv_cache
    path = env_file()
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return {}
    with _dotenv_lock:
        if _dotenv_cache and _dotenv_cache[0] == path and _dotenv_cache[1] == mtime:
            return _dotenv_cache[2]
        values = {k: v for k, v in dotenv_values(path).items() if v}
        _dotenv_cache = (path, mtime, values)
        return values


def get_env(name: str) -> str | None:
    """讀取環境變數；系統環境變數優先，其次是 .env 檔。"""
    if not name:
        return None
    return os.environ.get(name) or _dotenv().get(name) or None


def require_env(name: str) -> str:
    value = get_env(name)
    if not value:
        raise ValueError(f"環境變數 {name} 未設定，請在 .env 或系統環境變數中設定（範本見 .env.example）")
    return value


def _path_setting(name: str, default: Path) -> Path:
    value = get_env(name)
    if not value:
        return default
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def data_dir() -> Path:
    """所有執行期資料（資料集、向量快取、結果、SQLite）都放在這裡，預設為專案下的 data/。"""
    path = _path_setting("EMBEDBENCH_DATA_DIR", ROOT / "data")
    path.mkdir(parents=True, exist_ok=True)
    return path


def presets_dir() -> Path:
    return _path_setting("EMBEDBENCH_PRESETS_DIR", ROOT / "presets")


# 第三方套件（huggingface_hub、datasets、transformers）只讀系統環境變數，
# 啟動時把 .env 裡的這些設定帶進環境變數，才能下載需要授權的模型（例如 EmbeddingGemma）。
LIBRARY_ENV_VARS = ("HF_TOKEN", "HF_ENDPOINT", "HF_HOME")


def export_library_env() -> None:
    values = _dotenv()
    for name in LIBRARY_ENV_VARS:
        if not os.environ.get(name) and values.get(name):
            os.environ[name] = values[name]


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
