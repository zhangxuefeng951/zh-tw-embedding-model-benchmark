"""命令列工具：embedbench <指令>（或 python -m embedbench <指令>）。

常用範例：
  embedbench serve                                   # 啟動 Web UI（http://127.0.0.1:5000）
  embedbench presets                                 # 列出模型預設檔
  embedbench run --preset embeddinggemma-2 --base-url http://localhost:8000/v1 --dataset drcd-test
  embedbench run --config my-run.yaml                # 設定檔格式與 Web API 相同
  embedbench results --dataset drcd-test --format md
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import yaml

from embedbench import prefixes


def _unescape(s: str | None) -> str:
    """命令列上的 \\n 轉成換行（指令型前綴常用）。"""
    return (s or "").replace("\\n", "\n").replace("\\t", "\t")


def _parse_param(items: list[str] | None) -> dict:
    out = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--param 格式應為 key=value：{item}")
        k, v = item.split("=", 1)
        try:
            out[k] = json.loads(v)
        except ValueError:
            out[k] = v
    return out


class Progress:
    def __init__(self):
        self.last = 0.0
        self.stage = None
        self.tty = sys.stderr.isatty()

    def __call__(self, p: dict) -> None:
        now = time.monotonic()
        stage, done, total = p.get("stage"), p.get("done", 0), p.get("total", 0)
        bar = f" {done}/{total} ({done / total:.0%})" if total else ""
        if not self.tty:
            # 非終端機（例如寫進 log 檔）：每個階段只在開始與完成時各印一行
            finished = bool(total) and done >= total
            if stage != self.stage or (finished and now - self.last > 0):
                sys.stderr.write(f"{stage}{bar}\n")
                self.stage, self.last = stage, now if not finished else float("inf")
            return
        if stage != self.stage or now - self.last > 0.5 or (total and done >= total):
            if self.stage and stage != self.stage:
                sys.stderr.write("\n")
            self.stage, self.last = stage, now
            sys.stderr.write(f"\r{stage}{bar}\033[K")
            sys.stderr.flush()


def _print_results(result_ids: list[str], ks: list[int]) -> None:
    from embedbench.results import get_result, summarize

    rows = [summarize(get_result(r), ks) for r in result_ids]
    _print_table(rows, ks)


def _print_table(rows: list[dict], ks: list[int]) -> None:
    from embedbench.export import leaderboard_markdown

    if not rows:
        print("（沒有結果）")
        return
    print(leaderboard_markdown(rows, ks, ("hit", "mrr", "ndcg")))


def _run_job(kind: str, cfg: dict, ks: list[int]) -> None:
    from embedbench.jobs import run_job_sync

    sys.stderr.write("開始執行……\n")
    try:
        out = run_job_sync(kind, cfg, on_progress=Progress())
    except KeyboardInterrupt:
        sys.stderr.write("\n已中斷（已完成的向量已存入快取，重新執行會接續）\n")
        raise SystemExit(130) from None
    sys.stderr.write("\n")
    for m in out["messages"]:
        print(f"・{m}")
    _print_results(out["result_ids"], ks)


# ---- 指令 ---------------------------------------------------------------


def cmd_serve(args) -> None:
    from embedbench.web import create_app

    app = create_app(start_jobs=True)
    print(f"Web UI：http://{args.host}:{args.port}")
    if args.host not in ("127.0.0.1", "localhost"):
        print("注意：Web UI 沒有登入機制，開放到本機以外的網路前請自行評估風險。")
    app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=False, threaded=True)


def cmd_datasets(args) -> None:
    from embedbench import datasets

    if args.action == "list":
        for d in datasets.list_datasets():
            status = "已下載" if d["prepared"] else "未下載"
            size = f"{d.get('num_queries') or '?'} 題 / {d.get('num_documents') or '?'} 文件" if d["prepared"] else d["size_hint"]
            print(f"{d['key']:<32} {status:<4} {size:<24} {d['name']}")
    elif args.action == "prepare":
        for key in args.keys:
            ds = datasets.prepare_dataset(key, progress=lambda st, dn, t: Progress()({"stage": st, "done": dn, "total": t}), force=args.force)
            sys.stderr.write("\n")
            print(f"{key}：{len(ds.eval_queries)} 題、{len(ds.documents)} 文件")
    elif args.action == "import-simple":
        with open(args.file, "rb") as f:
            ds = datasets.import_simple(f.read(), args.file, args.name, args.description or "")
        print(f"已匯入 {ds.key}：{len(ds.eval_queries)} 題、{len(ds.documents)} 文件")
    elif args.action == "import-beir":
        if args.zip:
            with open(args.zip, "rb") as f:
                ds = datasets.import_beir(args.name, args.description or "", zip_content=f.read())
        else:
            if not (args.corpus and args.queries and args.qrels):
                raise SystemExit("BEIR 格式請提供 --zip，或同時提供 --corpus --queries --qrels")
            with open(args.corpus, "rb") as c, open(args.queries, "rb") as q, open(args.qrels, "rb") as r:
                ds = datasets.import_beir(args.name, args.description or "", c.read(), q.read(), r.read())
        print(f"已匯入 {ds.key}：{len(ds.eval_queries)} 題、{len(ds.documents)} 文件")
    elif args.action == "delete":
        for key in args.keys:
            datasets.delete_dataset(key)
            print(f"已刪除 {key}")


def cmd_presets(args) -> None:
    from embedbench.presets import list_presets

    for p in list_presets(args.kind):
        kind = p.get("provider") or p.get("reranker")
        print(f"{p['id']:<40} {kind:<22} {p['name']}")
        if args.verbose and args.kind == "embedding":
            for i, pair in enumerate(p["prefix_pairs"], start=1):
                mark = "*" if pair["default"] else " "
                print(
                    f"    {mark}{i}. {pair['label']}：Q={prefixes.display(pair['query'])!r} "
                    f"D={prefixes.display(pair['document'])!r}"
                )


def cmd_providers(args) -> None:
    from embedbench.providers import describe_providers
    from embedbench.rerankers import describe_rerankers

    items = describe_providers() if args.kind == "embedding" else describe_rerankers()
    for item in items:
        print(f"{item['type']}：{item['label']}")
        for p in item["params"]:
            print(f"    {p['name']:<22} 預設 {p['default']!r:<32} {p['label']}")


def _embedding_config_from_args(args) -> dict:
    from embedbench.presets import get_preset

    if args.config:
        with open(args.config, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {}
    preset = None
    if args.preset or cfg.get("preset_id"):
        preset = get_preset("embedding", args.preset or cfg["preset_id"])
        if not preset:
            raise SystemExit(f"找不到預設檔：{args.preset or cfg['preset_id']}（用 embedbench presets 查看）")
    provider = args.provider or cfg.get("provider") or (preset["provider"] if preset else "openai")
    params: dict = {}
    if preset:
        params.update(preset.get("params") or {})
        if provider != preset["provider"]:
            params = {"model": params.get("model")}
            params.update((preset.get("provider_params") or {}).get(provider) or {})
    params.update(cfg.get("params") or {})
    if args.base_url:
        params["base_url"] = args.base_url
    if args.model:
        params["model"] = args.model
    if args.api_key_env:
        params["api_key_env"] = args.api_key_env
    params.update(_parse_param(args.param))

    pairs = cfg.get("prefix_pairs")
    if args.query_prefix is not None or args.document_prefix is not None:
        pairs = [{"query": _unescape(args.query_prefix), "document": _unescape(args.document_prefix)}]
    elif pairs is None and preset:
        all_pairs = preset["prefix_pairs"]
        if args.all_pairs:
            pairs = all_pairs
        elif args.pair:
            pairs = []
            for sel in args.pair:
                if sel.isdigit() and 1 <= int(sel) <= len(all_pairs):
                    pairs.append(all_pairs[int(sel) - 1])
                else:
                    match = [p for p in all_pairs if sel in p["label"]]
                    if not match:
                        raise SystemExit(f"預設檔沒有符合「{sel}」的前綴組合")
                    pairs.extend(match)
        else:
            pairs = [p for p in all_pairs if p["default"]]
    return {
        "label": args.label or cfg.get("label") or "",
        "preset_id": preset["id"] if preset else None,
        "provider": provider,
        "params": params,
        "prefix_pairs": pairs or [],
        "datasets": args.dataset or cfg.get("datasets") or ["drcd-test"],
        "eval_dims": args.dims if args.dims is not None else cfg.get("eval_dims", []),
        "query_limit": args.limit if args.limit is not None else cfg.get("query_limit"),
    }


def _ks(value: str | None) -> list[int]:
    from embedbench.db import get_settings
    from embedbench.results import parse_ks

    return parse_ks(value) or get_settings()["default_ks"]


def cmd_run(args) -> None:
    cfg = _embedding_config_from_args(args)
    if args.dry_run:
        print(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
        return
    _run_job("embedding", cfg, _ks(args.k))


def cmd_test(args) -> None:
    from embedbench.runner import probe_embedding

    cfg = _embedding_config_from_args(args)
    out = probe_embedding(cfg["provider"], cfg["params"], cfg["prefix_pairs"], (cfg["datasets"] or [None])[0])
    print(f"載入 / 連線：{out['load_seconds']} 秒")
    for item in out["items"]:
        print(f"\n[{item['label']}] {item['dims']} 維，cosine={item['cosine']:.4f}，{item['elapsed_ms']} ms")
        print(f"  查詢送出：{item['query_text']!r}")
        print(f"  文件送出：{item['document_text'][:80]!r}…")
        print(f"  向量長度：查詢 {item['query_norm']:.4f}／文件 {item['document_norm']:.4f}")
    for w in out["warnings"]:
        print(f"⚠ {w}")


def cmd_bm25(args) -> None:
    cfg = {"datasets": args.dataset or ["drcd-test"], "k1": args.k1, "b": args.b, "query_limit": args.limit, "label": args.label}
    _run_job("bm25", cfg, _ks(args.k))


def cmd_rerank(args) -> None:
    from embedbench.presets import get_preset

    preset = get_preset("reranker", args.preset) if args.preset else None
    if args.preset and not preset:
        raise SystemExit(f"找不到 reranker 預設檔：{args.preset}（用 embedbench presets --kind reranker 查看）")
    reranker = args.reranker or (preset["reranker"] if preset else "rerank_api")
    params = dict(preset["params"]) if preset else {}
    if preset and reranker != preset["reranker"]:
        params = {"model": params.get("model"), **((preset.get("provider_params") or {}).get(reranker) or {})}
    if args.base_url:
        params["base_url"] = args.base_url
    if args.model:
        params["model"] = args.model
    params.update(_parse_param(args.param))
    cfg = {
        "source_result_id": args.source,
        "reranker": reranker,
        "params": params,
        "candidates": args.candidates or (preset or {}).get("candidates", 50),
        "include_title": args.include_title,
        "label": args.label,
        "preset_id": preset["id"] if preset else None,
    }
    _run_job("rerank", cfg, _ks(args.k))


def cmd_results(args) -> None:
    from embedbench import export
    from embedbench.results import list_results

    ks = _ks(args.k)
    rows = list_results(args.dataset, args.kind, ks)
    metric = args.sort if "@" in args.sort else f"{args.sort}@{5 if 5 in ks else ks[0]}"
    rows.sort(key=lambda r: r["metrics"].get(metric) or 0, reverse=True)
    if args.format == "csv":
        text = export.leaderboard_csv(rows, ks)
    elif args.format == "json":
        text = export.leaderboard_json(rows, ks)
    else:
        text = export.leaderboard_markdown(rows, ks, ("hit", "mrr", "ndcg"))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"已寫入 {args.output}")
    else:
        print(text)


def cmd_import_legacy(args) -> None:
    from embedbench.legacy import list_legacy_models

    models = list_legacy_models(args.path)
    print("舊資料庫中的模型：")
    for m in models:
        print(f"  {m['model']}：{m['questions']} 題、{m['paragraphs']} 段落")
    _run_job("legacy_import", {"path": args.path, "dataset": args.dataset, "models": args.model}, _ks(args.k))


def cmd_clear_cache(args) -> None:
    from embedbench.vectors import clear_cache

    for kind in ("vectors", "rerank", "bm25"):
        print(f"{kind}：清除 {clear_cache(kind)} 組快取")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="embedbench", description="繁體中文 Embedding / Reranker 檢索評測工具")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("serve", help="啟動 Web UI")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--debug", action="store_true")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("datasets", help="資料集：list / prepare / import-simple / import-beir / delete")
    dsub = p.add_subparsers(dest="action", required=True)
    dsub.add_parser("list", help="列出資料集")
    pp = dsub.add_parser("prepare", help="下載並轉換內建資料集")
    pp.add_argument("keys", nargs="+")
    pp.add_argument("--force", action="store_true", help="重新下載")
    pp = dsub.add_parser("import-simple", help="匯入簡單格式（CSV / JSONL：question, paragraph[, title]）")
    pp.add_argument("file")
    pp.add_argument("--name", required=True)
    pp.add_argument("--description")
    pp = dsub.add_parser("import-beir", help="匯入 BEIR 格式")
    pp.add_argument("--name", required=True)
    pp.add_argument("--description")
    pp.add_argument("--zip")
    pp.add_argument("--corpus")
    pp.add_argument("--queries")
    pp.add_argument("--qrels")
    pp = dsub.add_parser("delete", help="刪除自訂資料集，或清除內建資料集的下載快取")
    pp.add_argument("keys", nargs="+")
    p.set_defaults(func=cmd_datasets)

    p = sub.add_parser("presets", help="列出模型預設檔")
    p.add_argument("--kind", choices=["embedding", "reranker"], default="embedding")
    p.add_argument("-v", "--verbose", action="store_true", help="顯示前綴組合")
    p.set_defaults(func=cmd_presets)

    p = sub.add_parser("providers", help="列出 provider 與參數")
    p.add_argument("--kind", choices=["embedding", "reranker"], default="embedding")
    p.set_defaults(func=cmd_providers)

    for name, func, help_text in (
        ("run", cmd_run, "執行 embedding 評測"),
        ("test", cmd_test, "測試連線：送一筆查詢與文件，檢查維度、向量長度、NaN"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--config", help="YAML / JSON 設定檔（欄位與 Web API 相同）")
        p.add_argument("--preset", help="預設檔 id")
        p.add_argument("--provider", help="provider 類型（openai、cohere、voyage、gemini、vertex、sentence_transformers…）")
        p.add_argument("--base-url")
        p.add_argument("--model")
        p.add_argument("--api-key-env", help="API Key 的環境變數名稱")
        p.add_argument("--param", action="append", metavar="KEY=VALUE", help="其他 provider 參數，可重複")
        p.add_argument("--query-prefix", help="自訂查詢前綴（可用 \\n 表示換行）")
        p.add_argument("--document-prefix", help="自訂文件前綴（可用 {title}）")
        p.add_argument("--pair", action="append", help="選擇預設檔的前綴組合（編號或名稱關鍵字），可重複")
        p.add_argument("--all-pairs", action="store_true", help="使用預設檔的全部前綴組合")
        p.add_argument("--dataset", action="append", help="資料集 key，可重複（預設 drcd-test）")
        p.add_argument("--dims", type=lambda s: [int(x) for x in s.split(",") if x], help="評測維度，例如 768,512,256")
        p.add_argument("--limit", type=int, help="快速試跑：只跑前 N 題")
        p.add_argument("--label")
        p.add_argument("--k", help="顯示的 K 值，例如 1,3,5,10")
        p.add_argument("--dry-run", action="store_true", help="只印出最後的設定，不執行")
        p.set_defaults(func=func)

    p = sub.add_parser("bm25", help="執行 BM25 基準")
    p.add_argument("--dataset", action="append")
    p.add_argument("--k1", type=float, default=1.5)
    p.add_argument("--b", type=float, default=0.75)
    p.add_argument("--limit", type=int)
    p.add_argument("--label")
    p.add_argument("--k")
    p.set_defaults(func=cmd_bm25)

    p = sub.add_parser("rerank", help="執行 reranker 二階段評測")
    p.add_argument("--source", required=True, help="第一階段結果 ID（embedbench results 可查）")
    p.add_argument("--preset", help="reranker 預設檔 id")
    p.add_argument("--reranker", help="reranker 類型")
    p.add_argument("--base-url")
    p.add_argument("--model")
    p.add_argument("--param", action="append", metavar="KEY=VALUE")
    p.add_argument("--candidates", type=int, help="重新排序的候選數量（預設 50，最多 100）")
    p.add_argument("--include-title", action="store_true", help="文件前面加上標題")
    p.add_argument("--label")
    p.add_argument("--k")
    p.set_defaults(func=cmd_rerank)

    p = sub.add_parser("results", help="列出評測結果（排行榜）")
    p.add_argument("--dataset")
    p.add_argument("--kind", choices=["embedding", "bm25", "rerank", "legacy"])
    p.add_argument("--k")
    p.add_argument("--sort", default="hit", help="排序指標，例如 hit、mrr、ndcg@10")
    p.add_argument("--format", choices=["md", "csv", "json"], default="md")
    p.add_argument("--output", "-o")
    p.set_defaults(func=cmd_results)

    p = sub.add_parser("import-legacy", help="匯入舊版 data.db")
    p.add_argument("path")
    p.add_argument("--dataset", default="tceval-drcd-test", help="對應的資料集（預設 tceval-drcd-test，也可用 drcd-test）")
    p.add_argument("--model", action="append", help="只匯入指定模型，可重複")
    p.add_argument("--k")
    p.set_defaults(func=cmd_import_legacy)

    p = sub.add_parser("clear-cache", help="清除向量 / reranker / BM25 快取（不影響已完成的結果）")
    p.set_defaults(func=cmd_clear_cache)
    return parser


def main(argv: list[str] | None = None) -> None:
    from embedbench.config import export_library_env

    export_library_env()
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except (ValueError, KeyError, FileNotFoundError) as e:
        raise SystemExit(f"錯誤：{e}") from None
