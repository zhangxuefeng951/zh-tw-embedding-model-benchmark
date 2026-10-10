"""環境變數與 .env / .env.example。"""

import os

import yaml
from dotenv import dotenv_values

from embedbench import config


def write_env(path, text):
    path.write_text(text, encoding="utf-8")
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))  # 確保修改時間改變


def test_get_env_precedence_and_reload(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    monkeypatch.delenv("EB_TEST_KEY", raising=False)
    assert config.get_env("EB_TEST_KEY") is None
    write_env(env, "EB_TEST_KEY=from-file\nEB_EMPTY=\n")
    assert config.get_env("EB_TEST_KEY") == "from-file"
    assert config.get_env("EB_EMPTY") is None  # 空值視為未設定
    write_env(env, "EB_TEST_KEY=changed\n")
    assert config.get_env("EB_TEST_KEY") == "changed"  # 修改 .env 不必重啟
    monkeypatch.setenv("EB_TEST_KEY", "from-os")
    assert config.get_env("EB_TEST_KEY") == "from-os"  # 系統環境變數優先


def test_data_dir_from_dotenv(tmp_path, monkeypatch):
    monkeypatch.delenv("EMBEDBENCH_DATA_DIR")
    target = tmp_path / "custom-data"
    write_env(tmp_path / ".env", f"EMBEDBENCH_DATA_DIR={target}\n")
    assert config.data_dir() == target and target.is_dir()
    write_env(tmp_path / ".env", "EMBEDBENCH_DATA_DIR=relative/dir\n")
    assert config.data_dir() == config.ROOT / "relative/dir"
    (config.ROOT / "relative/dir").rmdir()
    (config.ROOT / "relative").rmdir()


def test_export_library_env(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    write_env(tmp_path / ".env", "HF_TOKEN=hf_file\nHF_ENDPOINT=https://mirror\nOPENAI_API_KEY=sk\n")
    monkeypatch.setenv("HF_ENDPOINT", "https://from-os")
    config.export_library_env()
    assert os.environ["HF_TOKEN"] == "hf_file"
    assert os.environ["HF_ENDPOINT"] == "https://from-os"  # 不覆蓋已設定的值
    assert "OPENAI_API_KEY" not in os.environ or os.environ["OPENAI_API_KEY"] != "sk"  # API Key 不放進環境變數


def test_env_example_covers_all_keys():
    """.env.example 要涵蓋設定頁列出的變數，以及所有內建預設檔用到的 API Key。"""
    example = dotenv_values(config.ROOT / ".env.example")
    assert all(v in (None, "") for v in example.values()), ".env.example 不應該包含實際的值"
    needed = set(config.KNOWN_ENV_VARS)
    for name in ("embedding.yaml", "reranker.yaml"):
        data = yaml.safe_load((config.ROOT / "presets" / name).read_text(encoding="utf-8"))
        for preset in data["presets"]:
            for params in [preset.get("params") or {}, *(preset.get("provider_params") or {}).values()]:
                if params.get("api_key_env"):
                    needed.add(params["api_key_env"])
    from embedbench.params import Param
    from embedbench.providers import PROVIDERS
    from embedbench.rerankers import RERANKERS

    for cls in [*PROVIDERS.values(), *RERANKERS.values()]:
        for p in cls.params:
            if isinstance(p, Param) and p.name.endswith("_env") and p.default:
                needed.add(p.default)
    missing = needed - set(example)
    assert not missing, f".env.example 缺少：{sorted(missing)}"
