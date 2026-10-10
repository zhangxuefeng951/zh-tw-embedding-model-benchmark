"""API 類 reranker：/v1/rerank 通用格式（Jina / Cohere / Voyage / vLLM / TEI），以及 LLM 列表式排序（RankGPT）。"""

from __future__ import annotations

import json
import re

import numpy as np

from embedbench.config import require_env
from embedbench.http import bearer, post_json
from embedbench.params import (
    Param,
    api_key_param,
    concurrency_param,
    model_param,
    retries_param,
    timeout_param,
)
from embedbench.providers.api import join_url
from embedbench.rerankers.base import Reranker


class _ApiReranker(Reranker):
    def api_key(self) -> str | None:
        name = (self.p.get("api_key_env") or "").strip()
        return require_env(name) if name else None

    def post(self, url: str, body: dict) -> dict | list:
        return post_json(
            url,
            body,
            headers=bearer(self.api_key()),
            timeout=float(self.p.get("timeout") or 120),
            max_retries=int(self.p.get("max_retries") or 0),
            check_cancel=self.check_cancel,
        )


class RerankApi(_ApiReranker):
    type_id = "rerank_api"
    label = "Rerank API（/v1/rerank）"
    group = "api"
    description = (
        "通用 rerank API。cohere 格式適用 Jina、Cohere v2、Voyage、vLLM 等（{model, query, documents} → "
        "results / data 的 relevance_score）；tei 格式適用 Hugging Face TEI（{query, texts} → score）。"
    )
    params = [
        Param("base_url", "API 端點（Base URL）", "text", "http://localhost:8000/v1", "系統會呼叫 {Base URL}/rerank"),
        model_param(),
        api_key_param(),
        Param("api_format", "API 格式", "select", "cohere", options=["cohere", "tei"]),
        concurrency_param(4),
        timeout_param(),
        retries_param(),
    ]

    def score(self, query: str, docs: list[str]) -> np.ndarray:
        url = join_url(self.p["base_url"], "rerank")
        if self.p["api_format"] == "tei":
            data = self.post(url, {"query": query, "texts": docs})
            items = data
        else:
            data = self.post(url, {"model": self.p["model"], "query": query, "documents": docs})
            items = data.get("results") if isinstance(data, dict) else data
            if items is None and isinstance(data, dict):
                items = data.get("data")
        if not isinstance(items, list):
            raise RuntimeError(f"rerank API 回應格式不符：{str(data)[:300]}")
        scores = np.full(len(docs), -np.inf, dtype=np.float64)
        for item in items:
            value = item.get("relevance_score", item.get("score"))
            scores[int(item["index"])] = float(value)
        return scores


RANKGPT_SYSTEM = "You are RankGPT, an intelligent assistant that can rank passages based on their relevancy to the query."


class LlmListwiseReranker(_ApiReranker):
    type_id = "llm_listwise"
    label = "LLM 列表式排序（RankGPT）"
    group = "llm"
    description = (
        "把候選段落編號後交給 LLM 排序（OpenAI 相容 chat/completions，可用 OpenAI、vLLM、Ollama、"
        "Gemini / Claude 的 OpenAI 相容端點等）。視窗大小 0＝一次排序全部候選（舊版 reranker-gpt.py 的做法）。"
    )
    params = [
        Param("base_url", "API 端點（Base URL）", "text", "https://api.openai.com/v1", "系統會呼叫 {Base URL}/chat/completions"),
        model_param("gpt-4o-mini"),
        api_key_param("OPENAI_API_KEY"),
        Param("window_size", "視窗大小", "int", 0, "0＝一次排序全部候選；RankGPT 論文常用 20"),
        Param("step", "滑動步長", "int", 10),
        Param("max_passage_chars", "段落最大字數", "int", 0, "0＝不截斷"),
        Param("temperature", "temperature", "float", 0.0),
        Param("json_mode", "要求 JSON 輸出", "bool", True, "送出 response_format={\"type\": \"json_object\"}"),
        concurrency_param(4),
        timeout_param(),
        retries_param(),
    ]

    def score(self, query: str, docs: list[str]) -> np.ndarray:
        n = len(docs)
        order = list(range(n))
        window = int(self.p["window_size"] or 0)
        step = max(1, int(self.p["step"] or 10))
        if window <= 0 or window >= n:
            order = self._permute(query, docs, order)
        else:
            end = n
            while True:
                start = max(end - window, 0)
                order[start:end] = self._permute(query, docs, order[start:end])
                if start == 0:
                    break
                end -= step
        scores = np.empty(n, dtype=np.float64)
        for pos, i in enumerate(order):
            scores[i] = n - pos
        return scores

    def _permute(self, query: str, docs: list[str], idx: list[int]) -> list[int]:
        num = len(idx)
        limit = int(self.p["max_passage_chars"] or 0)
        messages = [
            {"role": "system", "content": RANKGPT_SYSTEM},
            {
                "role": "user",
                "content": f"I will provide you with {num} passages, each indicated by number identifier []. \n"
                f"Rank the passages based on their relevance to query: {query}.",
            },
            {"role": "assistant", "content": "Okay, please provide the passages."},
        ]
        for rank, i in enumerate(idx, start=1):
            content = docs[i].strip()
            if limit > 0:
                content = content[:limit]
            messages.append({"role": "user", "content": f"[{rank}] {content}"})
            messages.append({"role": "assistant", "content": f"Received passage [{rank}]."})
        messages.append(
            {
                "role": "user",
                "content": f"Search Query: {query}. \nRank the {num} passages above based on their relevance to the "
                "search query. The passages should be listed in descending order using identifiers. The most "
                'relevant passages should be listed first. The output format should be a JSON object like '
                '{"ranking": [1, 3, 2]}. Only response the ranking results, do not say any word or explain.',
            }
        )
        body = {"model": self.p["model"], "messages": messages, "temperature": float(self.p["temperature"])}
        if self.p["json_mode"]:
            body["response_format"] = {"type": "json_object"}
        data = self.post(join_url(self.p["base_url"], "chat/completions"), body)
        content = data["choices"][0]["message"]["content"] or ""
        ranking = parse_ranking(content)
        seen, new = set(), []
        for r in ranking:
            p = r - 1
            if 0 <= p < num and p not in seen:
                seen.add(p)
                new.append(idx[p])
        new.extend(i for p, i in enumerate(idx) if p not in seen)
        return new


def parse_ranking(content: str) -> list[int]:
    """從 LLM 回覆取出排名：優先解析 JSON（物件中的第一個陣列或陣列本身），否則抓出所有數字。"""
    try:
        obj = json.loads(content)
        if isinstance(obj, dict):
            for v in obj.values():
                if isinstance(v, list):
                    obj = v
                    break
        if isinstance(obj, list):
            return [int(x) for x in obj if isinstance(x, (int, float)) or (isinstance(x, str) and x.strip().isdigit())]
    except (ValueError, TypeError):
        pass
    return [int(x) for x in re.findall(r"\d+", content)]
