"""透過 HTTP API 取得向量的 provider。"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time

import numpy as np

from embedbench.config import get_env, require_env
from embedbench.http import bearer, post_json
from embedbench.params import (
    Param,
    api_key_param,
    batch_size_param,
    concurrency_param,
    dimensions_param,
    extra_params,
    model_param,
    retries_param,
    timeout_param,
)
from embedbench.providers.base import EmbeddingProvider, parse_embedding_list
from embedbench.truncation import embed_with_truncation


def join_url(base: str, endpoint: str) -> str:
    """base 可以是 http://host/v1 或完整的 http://host/v1/embeddings。"""
    base = base.strip().rstrip("/")
    if base.endswith("/" + endpoint.strip("/")):
        return base
    return f"{base}/{endpoint.strip('/')}"


class _ApiProvider(EmbeddingProvider):
    group = "api"

    def api_key(self) -> str | None:
        name = (self.p.get("api_key_env") or "").strip()
        return require_env(name) if name else None

    def post(self, url: str, body: dict, headers: dict | None = None) -> dict | list:
        return post_json(
            url,
            body,
            headers=headers,
            timeout=float(self.p.get("timeout") or 120),
            max_retries=int(self.p.get("max_retries") or 0),
            check_cancel=self.check_cancel,
        )


class OpenAICompatProvider(_ApiProvider):
    type_id = "openai"
    label = "OpenAI 相容 API（/v1/embeddings）"
    description = (
        "任何相容 OpenAI /v1/embeddings 的服務：vLLM、TEI、Infinity、LiteRT-LM、Ollama、LM Studio、"
        "OpenAI、Mistral、Jina 等。系統只負責在文字前加上前綴並呼叫 API，模型服務請自行啟動。"
    )
    params = [
        Param(
            "base_url",
            "API 端點（Base URL）",
            "text",
            "http://localhost:8000/v1",
            "例如 http://localhost:8000/v1，系統會呼叫 {Base URL}/embeddings；也可以直接填完整的 .../v1/embeddings",
        ),
        model_param(),
        api_key_param(),
        dimensions_param("dimensions"),
        batch_size_param(32),
        concurrency_param(4),
        Param(
            "truncate",
            "超過長度上限時",
            "select",
            "auto",
            "auto＝服務端回報文字過長時，系統自動從尾端截短後重送（保留前綴與開頭，等同 sentence-transformers "
            "的預設截斷），並記錄截斷筆數；error＝直接報錯停止",
            options=["auto", "error"],
            affects_output=False,  # 沒超過上限的文字向量不受影響；截斷筆數另外記錄
        ),
        timeout_param(),
        retries_param(),
        *extra_params(),
    ]

    def embed_batch(self, texts: list[str], role: str) -> np.ndarray:
        if self.p.get("truncate") == "error":
            return self._request(texts, role)
        return embed_with_truncation(
            texts,
            lambda batch: self._request(batch, role),
            lambda original, kept: self.record_truncation(role, original, kept),
        )

    def _request(self, texts: list[str], role: str) -> np.ndarray:
        body = {"model": self.p["model"], "input": texts}
        if self.p.get("dimensions"):
            body["dimensions"] = int(self.p["dimensions"])
        body.update(self.role_extra(role))
        data = self.post(join_url(self.p["base_url"], "embeddings"), body, bearer(self.api_key()))
        out = parse_embedding_list(data.get("data") if isinstance(data, dict) else data, "/v1/embeddings")
        if len(out) != len(texts):
            raise RuntimeError(f"/v1/embeddings 回傳的向量數量不符：送出 {len(texts)} 筆，收到 {len(out)} 筆")
        return out


class CohereProvider(_ApiProvider):
    type_id = "cohere"
    label = "Cohere（原生 API）"
    description = "Cohere Embed v2 API，以 input_type 區分查詢與文件（search_query / search_document）。"
    params = [
        Param("base_url", "API 端點", "text", "https://api.cohere.com/v2", "系統會呼叫 {端點}/embed"),
        model_param("embed-v4.0"),
        api_key_param("COHERE_API_KEY"),
        Param("query_input_type", "查詢 input_type", "text", "search_query"),
        Param("document_input_type", "文件 input_type", "text", "search_document"),
        dimensions_param("output_dimension"),
        batch_size_param(96),
        concurrency_param(2),
        timeout_param(),
        retries_param(),
        *extra_params(),
    ]

    def embed_batch(self, texts: list[str], role: str) -> np.ndarray:
        input_type = self.p["query_input_type"] if role == "query" else self.p["document_input_type"]
        body = {"model": self.p["model"], "texts": texts, "embedding_types": ["float"]}
        if input_type:
            body["input_type"] = input_type
        if self.p.get("dimensions"):
            body["output_dimension"] = int(self.p["dimensions"])
        body.update(self.role_extra(role))
        data = self.post(join_url(self.p["base_url"], "embed"), body, bearer(self.api_key()))
        embeddings = data.get("embeddings")
        if isinstance(embeddings, dict):
            embeddings = embeddings.get("float")
        if embeddings is None:
            raise RuntimeError(f"Cohere 回應格式不符：{str(data)[:300]}")
        return np.asarray(embeddings, dtype=np.float32)


class VoyageProvider(_ApiProvider):
    type_id = "voyage"
    label = "Voyage AI（原生 API）"
    description = "Voyage Embeddings API，以 input_type 區分查詢與文件（query / document）。也支援 multimodal 端點。"
    params = [
        Param("base_url", "API 端點", "text", "https://api.voyageai.com/v1"),
        Param(
            "endpoint",
            "端點類型",
            "select",
            "embeddings",
            "voyage-multimodal-3 等多模態模型請選 multimodalembeddings",
            options=["embeddings", "multimodalembeddings"],
        ),
        model_param("voyage-3.5"),
        api_key_param("VOYAGE_API_KEY"),
        Param("query_input_type", "查詢 input_type", "text", "query", "留空＝不送 input_type"),
        Param("document_input_type", "文件 input_type", "text", "document", "留空＝不送 input_type"),
        dimensions_param("output_dimension"),
        batch_size_param(64),
        concurrency_param(2),
        timeout_param(),
        retries_param(),
        *extra_params(),
    ]

    def embed_batch(self, texts: list[str], role: str) -> np.ndarray:
        input_type = self.p["query_input_type"] if role == "query" else self.p["document_input_type"]
        if self.p["endpoint"] == "multimodalembeddings":
            body = {
                "model": self.p["model"],
                "inputs": [{"content": [{"type": "text", "text": t}]} for t in texts],
            }
        else:
            body = {"model": self.p["model"], "input": texts}
        if input_type:
            body["input_type"] = input_type
        if self.p.get("dimensions"):
            body["output_dimension"] = int(self.p["dimensions"])
        body.update(self.role_extra(role))
        data = self.post(join_url(self.p["base_url"], self.p["endpoint"]), body, bearer(self.api_key()))
        return parse_embedding_list(data.get("data"), "Voyage")


class GeminiProvider(_ApiProvider):
    type_id = "gemini"
    label = "Google Gemini API（原生）"
    description = "Gemini API batchEmbedContents，以 taskType 區分查詢與文件（RETRIEVAL_QUERY / RETRIEVAL_DOCUMENT）。"
    params = [
        Param("base_url", "API 端點", "text", "https://generativelanguage.googleapis.com/v1beta"),
        model_param("gemini-embedding-001"),
        api_key_param("GEMINI_API_KEY"),
        Param("query_task_type", "查詢 taskType", "text", "RETRIEVAL_QUERY", "留空＝不送 taskType"),
        Param("document_task_type", "文件 taskType", "text", "RETRIEVAL_DOCUMENT", "留空＝不送 taskType"),
        dimensions_param("outputDimensionality"),
        batch_size_param(100),
        concurrency_param(2),
        timeout_param(),
        retries_param(),
        *extra_params(),
    ]

    def embed_batch(self, texts: list[str], role: str) -> np.ndarray:
        model = self.p["model"]
        model_path = model if model.startswith("models/") else f"models/{model}"
        task_type = self.p["query_task_type"] if role == "query" else self.p["document_task_type"]
        extra = self.role_extra(role)
        requests_ = []
        for t in texts:
            r = {"model": model_path, "content": {"parts": [{"text": t}]}}
            if task_type:
                r["taskType"] = task_type
            if self.p.get("dimensions"):
                r["outputDimensionality"] = int(self.p["dimensions"])
            r.update(extra)
            requests_.append(r)
        url = join_url(self.p["base_url"], f"{model_path}:batchEmbedContents")
        headers = {"x-goog-api-key": self.api_key() or ""}
        data = self.post(url, {"requests": requests_}, headers)
        return np.asarray([e["values"] for e in data["embeddings"]], dtype=np.float32)


class VertexProvider(_ApiProvider):
    type_id = "vertex"
    label = "Google Vertex AI（原生）"
    description = (
        "Vertex AI text embedding（predict API），以 task_type 區分查詢與文件。"
        "存取權杖讀取環境變數；未設定時會嘗試執行 gcloud auth print-access-token。"
    )
    params = [
        model_param("text-multilingual-embedding-002"),
        Param("project", "GCP 專案 ID", "text", "", "留空＝讀取環境變數 GCP_PROJECT_ID", affects_output=False),
        Param("location", "區域", "text", "us-central1"),
        Param(
            "access_token_env",
            "存取權杖環境變數名稱",
            "text",
            "GCP_ACCESS_TOKEN",
            "未設定時改用 gcloud auth print-access-token",
            affects_output=False,
        ),
        Param("query_task_type", "查詢 task_type", "text", "RETRIEVAL_QUERY", "留空＝不送 task_type"),
        Param("document_task_type", "文件 task_type", "text", "RETRIEVAL_DOCUMENT", "留空＝不送 task_type"),
        dimensions_param("outputDimensionality"),
        batch_size_param(5),
        concurrency_param(2),
        timeout_param(),
        retries_param(),
        *extra_params(),
    ]

    _token_lock = threading.Lock()
    _token_cache: tuple[str, float] | None = None

    def _token(self) -> str:
        token = get_env(self.p.get("access_token_env") or "")
        if token:
            return token
        with self._token_lock:
            cached = VertexProvider._token_cache
            if cached and time.time() - cached[1] < 30 * 60:
                return cached[0]
            gcloud = shutil.which("gcloud")
            if not gcloud:
                raise ValueError("找不到 Vertex AI 存取權杖：請設定 GCP_ACCESS_TOKEN 或安裝 gcloud")
            token = subprocess.run(
                [gcloud, "auth", "print-access-token"], capture_output=True, text=True, check=True, timeout=60
            ).stdout.strip()
            VertexProvider._token_cache = (token, time.time())
            return token

    def embed_batch(self, texts: list[str], role: str) -> np.ndarray:
        project = self.p.get("project") or require_env("GCP_PROJECT_ID")
        location = self.p["location"]
        task_type = self.p["query_task_type"] if role == "query" else self.p["document_task_type"]
        extra = self.role_extra(role)
        instances = []
        for t in texts:
            inst = {"content": t}
            if task_type:
                inst["task_type"] = task_type
            inst.update(extra)
            instances.append(inst)
        body: dict = {"instances": instances}
        if self.p.get("dimensions"):
            body["parameters"] = {"outputDimensionality": int(self.p["dimensions"])}
        url = (
            f"https://{location}-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}"
            f"/publishers/google/models/{self.p['model']}:predict"
        )
        data = self.post(url, body, bearer(self._token()))
        return np.asarray([p["embeddings"]["values"] for p in data["predictions"]], dtype=np.float32)
