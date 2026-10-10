"""本地 reranker：CrossEncoder、FlagEmbedding（BGE）、Qwen3-Reranker。需要 uv sync --extra local"""

from __future__ import annotations

import numpy as np

from embedbench.params import Param, batch_size_param, model_param
from embedbench.providers.local import _free_memory, device_param, dtype_kwargs, require, resolve_device
from embedbench.rerankers.base import Reranker


class CrossEncoderReranker(Reranker):
    type_id = "cross_encoder"
    label = "CrossEncoder（sentence-transformers，本地）"
    group = "local"
    description = "sentence-transformers 的 CrossEncoder，例如 cross-encoder/ms-marco-MiniLM-L-6-v2。"
    params = [
        model_param("cross-encoder/ms-marco-MiniLM-L-6-v2"),
        Param("max_length", "最大 token 數", "int", 512),
        device_param(),
        Param("trust_remote_code", "trust_remote_code", "bool", False),
        batch_size_param(32),
    ]

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        st = require("sentence_transformers")
        self.model = st.CrossEncoder(
            self.p["model"],
            max_length=int(self.p["max_length"]),
            device=resolve_device(self.p["device"]),
            trust_remote_code=bool(self.p["trust_remote_code"]),
        )

    def score(self, query: str, docs: list[str]) -> np.ndarray:
        pairs = [(query, d) for d in docs]
        out = self.model.predict(pairs, batch_size=int(self.p["batch_size"]), show_progress_bar=False)
        return np.asarray(out, dtype=np.float64).reshape(len(docs))

    def close(self) -> None:
        del self.model
        _free_memory()


class FlagReranker(Reranker):
    type_id = "flag_reranker"
    label = "FlagEmbedding Reranker（BGE，本地）"
    group = "local"
    description = "FlagEmbedding 的 BGE reranker（bge-reranker-v2-m3 等）；LLM 型請選 FlagLLMReranker / LayerWiseFlagLLMReranker。"
    params = [
        model_param("BAAI/bge-reranker-v2-m3"),
        Param(
            "reranker_class",
            "Reranker 類別",
            "select",
            "FlagReranker",
            options=["FlagReranker", "FlagLLMReranker", "LayerWiseFlagLLMReranker"],
        ),
        Param("use_fp16", "使用 FP16", "bool", False, "GPU 上可加速；CPU 請關閉"),
        Param("score_kwargs", "compute_score 額外參數（JSON）", "json", {}, '例如 LayerWise 模型：{"cutoff_layers": [28]}', advanced=True),
        batch_size_param(32),
    ]

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        flag = require("FlagEmbedding")
        cls = getattr(flag, self.p["reranker_class"])
        self.model = cls(self.p["model"], use_fp16=bool(self.p["use_fp16"]))

    def score(self, query: str, docs: list[str]) -> np.ndarray:
        pairs = [[query, d] for d in docs]
        out = self.model.compute_score(pairs, batch_size=int(self.p["batch_size"]), **(self.p["score_kwargs"] or {}))
        if isinstance(out, (int, float)):
            out = [out]
        return np.asarray(out, dtype=np.float64).reshape(len(docs))

    def close(self) -> None:
        del self.model
        _free_memory()


QWEN3_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct "
    'provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
)
QWEN3_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


class Qwen3Reranker(Reranker):
    type_id = "qwen3_reranker"
    label = "Qwen3-Reranker（transformers，本地）"
    group = "local"
    description = "Qwen3-Reranker 系列：以 LLM 判斷 yes / no 的機率作為相關分數。"
    params = [
        model_param("Qwen/Qwen3-Reranker-0.6B"),
        Param(
            "instruction",
            "任務指令",
            "text",
            "Given a web search query, retrieve relevant passages that answer the query",
        ),
        Param("max_length", "最大 token 數", "int", 8192),
        device_param(),
        Param("dtype", "計算精度", "select", "auto", options=["auto", "float32", "bfloat16", "float16"]),
        batch_size_param(8),
    ]

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        torch = require("torch")
        transformers = require("transformers")
        device = resolve_device(self.p["device"])
        if device is None:
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.torch, self.device = torch, device
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.p["model"], padding_side="left")
        self.model = (
            transformers.AutoModelForCausalLM.from_pretrained(self.p["model"], **dtype_kwargs(self.p["dtype"]))
            .to(device)
            .eval()
        )
        self.token_false = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true = self.tokenizer.convert_tokens_to_ids("yes")
        self.prefix_tokens = self.tokenizer.encode(QWEN3_PREFIX, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(QWEN3_SUFFIX, add_special_tokens=False)

    def score(self, query: str, docs: list[str]) -> np.ndarray:
        torch = self.torch
        max_length = int(self.p["max_length"])
        texts = [f"<Instruct>: {self.p['instruction']}\n<Query>: {query}\n<Document>: {d}" for d in docs]
        out: list[float] = []
        bs = max(1, int(self.p["batch_size"]))
        for i in range(0, len(texts), bs):
            if self.check_cancel:
                self.check_cancel()
            enc = self.tokenizer(
                texts[i : i + bs],
                padding=False,
                truncation="longest_first",
                return_attention_mask=False,
                max_length=max_length - len(self.prefix_tokens) - len(self.suffix_tokens),
            )
            enc["input_ids"] = [self.prefix_tokens + ids + self.suffix_tokens for ids in enc["input_ids"]]
            batch = self.tokenizer.pad(enc, padding=True, return_tensors="pt", max_length=max_length)
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                logits = self.model(**batch).logits[:, -1, :]
                pair = torch.stack([logits[:, self.token_false], logits[:, self.token_true]], dim=1)
                probs = torch.nn.functional.log_softmax(pair.float(), dim=1)[:, 1].exp()
            out.extend(probs.cpu().tolist())
        return np.asarray(out, dtype=np.float64)

    def close(self) -> None:
        del self.model
        _free_memory()
