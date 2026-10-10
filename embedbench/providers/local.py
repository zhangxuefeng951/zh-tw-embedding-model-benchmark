"""由系統直接載入的本地模型。需要先安裝選用套件：uv sync --extra local"""

from __future__ import annotations

import numpy as np

from embedbench.params import Param, batch_size_param, dimensions_param, model_param
from embedbench.providers.base import EmbeddingProvider

INSTALL_HINT = "本地模型需要額外套件，請執行：uv sync --extra local"


def require(module: str):
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as e:
        raise RuntimeError(f"缺少 {module}。{INSTALL_HINT}") from e


def device_param() -> Param:
    return Param(
        "device",
        "裝置",
        "select",
        "auto",
        "auto＝自動選擇（CUDA > MPS > CPU）",
        options=["auto", "cpu", "cuda", "mps"],
        affects_output=False,
    )


def dtype_param() -> Param:
    return Param(
        "dtype",
        "計算精度",
        "select",
        "auto",
        "EmbeddingGemma 系列請用 bfloat16 或 float32，不要用 float16（可能產生 NaN 或降低品質）",
        options=["auto", "float32", "bfloat16", "float16"],
    )


def torch_dtype(name: str):
    if not name or name == "auto":
        return None
    torch = require("torch")
    return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[name]


def dtype_kwargs(name: str) -> dict:
    """transformers 4.56 起把 torch_dtype 改名為 dtype。"""
    dt = torch_dtype(name)
    if dt is None:
        return {}
    transformers = require("transformers")
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    return {"dtype": dt} if (major, minor) >= (4, 56) else {"torch_dtype": dt}


def resolve_device(name: str) -> str | None:
    return None if not name or name == "auto" else name


class SentenceTransformersProvider(EmbeddingProvider):
    type_id = "sentence_transformers"
    label = "sentence-transformers（本地）"
    group = "local"
    description = (
        "以 sentence-transformers 載入 Hugging Face 模型。前綴由本系統加上；"
        "系統會關閉套件自動套用的預設 prompt，避免重複加前綴（除非你在額外參數指定 prompt_name）。"
    )
    params = [
        model_param("intfloat/multilingual-e5-large"),
        device_param(),
        dtype_param(),
        Param("trust_remote_code", "trust_remote_code", "bool", False, "部分模型（Jina、Nomic、NV-Embed）需要開啟"),
        Param("max_seq_length", "最大序列長度", "int", None, "留空＝模型預設", placeholder="留空＝預設"),
        Param(
            "append_eos",
            "在文字後加上 EOS 標記",
            "bool",
            False,
            "NV-Embed-v2 需要在每段文字後面補上 tokenizer 的 EOS 標記",
        ),
        dimensions_param("truncate_dim"),
        Param(
            "load_kwargs",
            "載入參數（JSON）",
            "json",
            {},
            "傳給 SentenceTransformer(...) 的額外參數，例如 EmbeddingGemma 2 只載入文字模組："
            '{"config_kwargs": {"vision_config": null, "audio_config": null}}',
            advanced=True,
        ),
        Param(
            "query_extra",
            "查詢 encode 參數（JSON）",
            "json",
            {},
            '傳給 model.encode() 的額外參數，例如 Jina v4：{"task": "retrieval", "prompt_name": "query"}',
            advanced=True,
        ),
        Param("document_extra", "文件 encode 參數（JSON）", "json", {}, advanced=True),
        batch_size_param(32),
    ]

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        st = require("sentence_transformers")
        kwargs = dict(self.p.get("load_kwargs") or {})
        model_kwargs = dict(kwargs.pop("model_kwargs", {}) or {})
        model_kwargs.update(dtype_kwargs(self.p["dtype"]))
        if model_kwargs:
            kwargs["model_kwargs"] = model_kwargs
        if self.p.get("dimensions"):
            kwargs["truncate_dim"] = int(self.p["dimensions"])
        self.model = st.SentenceTransformer(
            self.p["model"],
            device=resolve_device(self.p["device"]),
            trust_remote_code=bool(self.p["trust_remote_code"]),
            **kwargs,
        )
        if self.p.get("max_seq_length"):
            self.model.max_seq_length = int(self.p["max_seq_length"])
        if self.p.get("append_eos"):
            self.model.tokenizer.padding_side = "right"
        # 前綴由本系統負責；關掉模型設定的預設 prompt，避免重複加前綴
        if getattr(self.model, "default_prompt_name", None):
            self.model.default_prompt_name = None

    def embed_batch(self, texts: list[str], role: str) -> np.ndarray:
        extra = self.role_extra(role)
        if self.p.get("append_eos"):
            texts = [t + self.model.tokenizer.eos_token for t in texts]
        out = self.model.encode(
            texts,
            batch_size=len(texts),
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
            **extra,
        )
        return np.asarray(out, dtype=np.float32).reshape(len(texts), -1)

    def close(self) -> None:
        del self.model
        _free_memory()


class TransformersProvider(EmbeddingProvider):
    type_id = "transformers"
    label = "transformers（本地，CLS / mean pooling）"
    group = "local"
    description = "直接用 transformers 的 AutoModel 取隱藏層向量，適合沒有 sentence-transformers 設定的模型（例如 ckiplab/bert-base-chinese）。"
    params = [
        model_param("ckiplab/bert-base-chinese"),
        Param("pooling", "Pooling", "select", "cls", "cls＝取第一個 token；mean＝平均所有 token", options=["cls", "mean"]),
        Param("max_length", "最大 token 數", "int", 512, "超過會截斷"),
        device_param(),
        dtype_param(),
        Param("trust_remote_code", "trust_remote_code", "bool", False),
        batch_size_param(16),
    ]

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        torch = require("torch")
        transformers = require("transformers")
        device = resolve_device(self.p["device"])
        if device is None:
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.device = device
        trust = bool(self.p["trust_remote_code"])
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.p["model"], trust_remote_code=trust)
        self.model = transformers.AutoModel.from_pretrained(
            self.p["model"], trust_remote_code=trust, **dtype_kwargs(self.p["dtype"])
        ).to(device)
        self.model.eval()
        self.torch = torch

    def embed_batch(self, texts: list[str], role: str) -> np.ndarray:
        torch = self.torch
        inputs = self.tokenizer(
            texts, max_length=int(self.p["max_length"]), truncation=True, padding=True, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            hidden = self.model(**inputs).last_hidden_state
        if self.p["pooling"] == "mean":
            mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = hidden[:, 0]
        return pooled.float().cpu().numpy()

    def close(self) -> None:
        del self.model
        _free_memory()


class FlagEmbeddingProvider(EmbeddingProvider):
    type_id = "flag_embedding"
    label = "FlagEmbedding（本地）"
    group = "local"
    description = "以 FlagEmbedding 的 FlagModel 載入（例如 chuxin-llm/Chuxin-Embedding、BGE 系列）。"
    params = [
        model_param("chuxin-llm/Chuxin-Embedding"),
        Param("use_fp16", "使用 FP16", "bool", False, "GPU 上可加速；CPU 請關閉"),
        Param("max_length", "最大 token 數", "int", 512),
        batch_size_param(32),
    ]

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        flag = require("FlagEmbedding")
        self.model = flag.FlagModel(self.p["model"], use_fp16=bool(self.p["use_fp16"]))

    def embed_batch(self, texts: list[str], role: str) -> np.ndarray:
        out = self.model.encode(texts, batch_size=len(texts), max_length=int(self.p["max_length"]))
        return np.asarray(out, dtype=np.float32).reshape(len(texts), -1)

    def close(self) -> None:
        del self.model
        _free_memory()


class Model2VecProvider(EmbeddingProvider):
    type_id = "model2vec"
    label = "Model2Vec 靜態向量（本地）"
    group = "local"
    description = "Model2Vec 靜態嵌入模型（例如 minishlab/potion-base-32M），CPU 即可快速執行。"
    params = [model_param("minishlab/potion-base-32M"), batch_size_param(256)]

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        m2v = require("model2vec")
        self.model = m2v.StaticModel.from_pretrained(self.p["model"])

    def embed_batch(self, texts: list[str], role: str) -> np.ndarray:
        return np.asarray(self.model.encode(texts), dtype=np.float32).reshape(len(texts), -1)


def _free_memory() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
