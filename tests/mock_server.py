"""測試用的假 API 伺服器：/v1/embeddings、/v1/rerank、/v1/chat/completions。

向量以字元 bigram 雜湊產生（相同字詞越多越相近），所以檢索結果有意義；
伺服器會記錄收到的每一筆文字，測試可以檢查前綴有沒有正確加上。

手動啟動：python tests/mock_server.py --port 8765
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np


def fake_embedding(text: str, dims: int = 256) -> np.ndarray:
    v = np.zeros(dims, dtype=np.float64)
    chars = [c for c in text if not c.isspace()]
    grams = chars + [a + b for a, b in zip(chars, chars[1:])]
    for g in grams:
        h = int.from_bytes(hashlib.md5(g.encode()).digest()[:8], "little")
        v[h % dims] += 1.0 if (h >> 32) & 1 else -1.0
    n = np.linalg.norm(v)
    return v / n if n else v


class MockState:
    def __init__(self):
        self.embedding_inputs: list[str] = []
        self.requests: list[dict] = []
        self.fail_next = 0  # 前 N 個請求回 503（測試重試）
        self.dims = 256
        self.lock = threading.Lock()


def make_handler(state: MockState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code: int, obj) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            with state.lock:
                state.requests.append({"path": self.path, "body": body, "auth": self.headers.get("Authorization")})
                if state.fail_next > 0:
                    state.fail_next -= 1
                    return self._send(503, {"error": "temporarily unavailable"})
            if self.path.endswith("/embeddings"):
                inputs = body.get("input")
                if isinstance(inputs, str):
                    inputs = [inputs]
                dims = int(body.get("dimensions") or state.dims)
                with state.lock:
                    state.embedding_inputs.extend(inputs)
                data = [
                    {"object": "embedding", "index": i, "embedding": fake_embedding(t, state.dims)[:dims].tolist()}
                    for i, t in enumerate(inputs)
                ]
                return self._send(200, {"object": "list", "data": data, "model": body.get("model")})
            if self.path.endswith("/rerank"):
                q = fake_embedding(body["query"])
                docs = body.get("documents") or body.get("texts")
                results = [
                    {"index": i, "relevance_score": float(q @ fake_embedding(d))} for i, d in enumerate(docs)
                ]
                results.sort(key=lambda x: -x["relevance_score"])
                return self._send(200, {"results": results})
            if self.path.endswith("/chat/completions"):
                msgs = body["messages"]
                query = re.search(r"Search Query: (.*)\. \n", msgs[-1]["content"]).group(1)
                passages = [m["content"] for m in msgs if m["role"] == "user" and re.match(r"^\[\d+\] ", m["content"])]
                q = fake_embedding(query)
                scored = sorted(
                    range(len(passages)), key=lambda i: -float(q @ fake_embedding(passages[i].split("] ", 1)[1]))
                )
                content = json.dumps({"ranking": [i + 1 for i in scored]})
                return self._send(200, {"choices": [{"message": {"role": "assistant", "content": content}}]})
            return self._send(404, {"error": "not found"})

    return Handler


def start(port: int = 0) -> tuple[ThreadingHTTPServer, MockState, str]:
    state = MockState()
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(state))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, state, f"http://127.0.0.1:{server.server_address[1]}/v1"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server, _, url = start(args.port)
    print(f"mock API: {url}")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        server.shutdown()
