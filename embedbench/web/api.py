"""JSON API（Web UI 使用，也可以直接用 curl 呼叫）。"""

from __future__ import annotations

import traceback

from flask import Blueprint, Response, current_app, jsonify, request

from embedbench import config, datasets, export, presets
from embedbench.db import DEFAULT_SETTINGS, get_db, get_settings
from embedbench.jobs import JobManager
from embedbench.providers import describe_providers
from embedbench.rerankers import describe_rerankers
from embedbench.results import (
    clean,
    delete_result,
    get_result,
    k_curve,
    list_results,
    parse_ks,
    query_details,
    result_detail,
)

api = Blueprint("api", __name__, url_prefix="/api")


def jobs() -> JobManager:
    return current_app.extensions["jobs"]


def body() -> dict:
    return request.get_json(silent=True) or {}


def ks_arg() -> list[int]:
    return parse_ks(request.args.get("ks")) or get_settings()["default_ks"]


@api.errorhandler(ValueError)
def bad_request(e):
    return jsonify({"error": str(e)}), 400


@api.errorhandler(KeyError)
def not_found(e):
    return jsonify({"error": f"找不到：{e.args[0] if e.args else ''}"}), 404


@api.errorhandler(FileNotFoundError)
def file_not_found(e):
    return jsonify({"error": str(e)}), 404


@api.errorhandler(Exception)
def server_error(e):
    if isinstance(getattr(e, "code", None), int):
        return jsonify({"error": str(e)}), e.code
    current_app.logger.error(traceback.format_exc())
    return jsonify({"error": f"{e.__class__.__name__}：{e}"}), 500


# ---- 中繼資料 -------------------------------------------------------------


@api.get("/providers")
def providers():
    kind = request.args.get("kind", "embedding")
    return jsonify(describe_providers() if kind == "embedding" else describe_rerankers())


@api.get("/env")
def env_status():
    names = list(config.KNOWN_ENV_VARS)
    for n in request.args.getlist("name"):
        if n and n not in names:
            names.append(n)
    return jsonify({n: bool(config.get_env(n)) for n in names})


@api.get("/settings")
def get_settings_api():
    return jsonify({**get_settings(), "data_dir": str(config.data_dir()), "presets_dir": str(config.presets_dir())})


@api.put("/settings")
def put_settings():
    data = body()
    db = get_db()
    if "max_concurrent_jobs" in data:
        n = int(data["max_concurrent_jobs"])
        if not 1 <= n <= 16:
            raise ValueError("同時執行的任務數必須介於 1～16")
        db.set_setting("max_concurrent_jobs", n)
    if "default_ks" in data:
        ks = parse_ks(data["default_ks"])
        if not ks or max(ks) > 100:
            raise ValueError("K 值必須是 1～100 的整數，以逗號分隔")
        db.set_setting("default_ks", ks)
    unknown = set(data) - set(DEFAULT_SETTINGS)
    if unknown:
        raise ValueError(f"未知的設定：{', '.join(unknown)}")
    jobs()._wake.set()
    return jsonify(get_settings())


@api.post("/cache/clear")
def clear_cache():
    from embedbench.vectors import clear_cache as clear

    return jsonify({kind: clear(kind) for kind in ("vectors", "rerank", "bm25")})


# ---- 預設檔 -------------------------------------------------------------


@api.get("/presets")
def list_presets_api():
    return jsonify(presets.list_presets(request.args.get("kind", "embedding")))


@api.post("/presets")
def save_preset_api():
    data = body()
    preset = presets.save_user_preset(data.get("kind", "embedding"), data.get("preset") or {}, data.get("id"))
    return jsonify(preset)


@api.delete("/presets/<preset_id>")
def delete_preset_api(preset_id: str):
    presets.delete_user_preset(preset_id)
    return jsonify({"ok": True})


@api.get("/presets/<preset_id>/yaml")
def preset_yaml(preset_id: str):
    preset = presets.get_preset(request.args.get("kind", "embedding"), preset_id)
    if not preset:
        raise KeyError(preset_id)
    return Response(presets.export_yaml(preset), mimetype="text/plain; charset=utf-8")


# ---- 資料集 -------------------------------------------------------------


@api.get("/datasets")
def list_datasets_api():
    return jsonify(datasets.list_datasets())


@api.get("/datasets/<key>/sample")
def dataset_sample(key: str):
    ds = datasets.get_dataset(key, prepare=False)
    n = min(int(request.args.get("n", 3)), 20)
    items = []
    for q in ds.eval_queries[:n]:
        gold = [ds.documents[i] for i in ds.relevant(q.id)]
        items.append(
            {"query_id": q.id, "query": q.text, "gold": [{"id": d.id, "title": d.title, "text": d.text} for d in gold]}
        )
    return jsonify({**ds.summary(), "meta": ds.meta, "items": items})


@api.post("/datasets/<key>/prepare")
def dataset_prepare(key: str):
    job = jobs().submit("prepare_dataset", {"dataset": key, "force": bool(body().get("force"))})
    return jsonify(job)


@api.delete("/datasets/<key>")
def dataset_delete(key: str):
    datasets.delete_dataset(key)
    return jsonify({"ok": True})


@api.post("/datasets/upload")
def dataset_upload():
    fmt = request.form.get("format", "simple")
    name = (request.form.get("name") or "").strip()
    description = request.form.get("description") or ""
    if not name:
        raise ValueError("請填寫資料集名稱")
    files = request.files
    if fmt == "simple":
        f = files.get("file")
        if not f or not f.filename:
            raise ValueError("請選擇檔案")
        ds = datasets.import_simple(f.read(), f.filename, name, description)
    elif fmt == "beir":
        if files.get("zip") and files["zip"].filename:
            ds = datasets.import_beir(name, description, zip_content=files["zip"].read())
        else:
            parts = {}
            for k in ("corpus", "queries", "qrels"):
                f = files.get(k)
                if not f or not f.filename:
                    raise ValueError("BEIR 格式請上傳 zip，或同時上傳 corpus.jsonl、queries.jsonl、qrels.tsv")
                parts[k] = f.read()
            ds = datasets.import_beir(name, description, **parts)
    else:
        raise ValueError(f"未知的格式：{fmt}")
    return jsonify(ds.summary())


# ---- 連線測試 ------------------------------------------------------------


@api.post("/test/embedding")
def test_embedding_api():
    from embedbench.runner import probe_embedding

    d = body()
    try:
        return jsonify(
            clean(probe_embedding(d.get("provider") or "openai", d.get("params") or {}, d.get("prefix_pairs") or [], d.get("dataset")))
        )
    except (ValueError, KeyError):
        raise
    except Exception as e:
        return jsonify({"ok": False, "error": f"{e.__class__.__name__}：{e}"}), 200


@api.post("/test/reranker")
def test_reranker_api():
    from embedbench.runner import probe_reranker

    d = body()
    try:
        return jsonify(clean(probe_reranker(d.get("reranker") or "rerank_api", d.get("params") or {}, d.get("dataset"))))
    except (ValueError, KeyError):
        raise
    except Exception as e:
        return jsonify({"ok": False, "error": f"{e.__class__.__name__}：{e}"}), 200


# ---- 任務 ---------------------------------------------------------------


@api.get("/jobs")
def list_jobs():
    return jsonify(clean(jobs().list(int(request.args.get("limit", 100)))))


@api.get("/jobs/active")
def active_jobs():
    rows = get_db().query("SELECT status, COUNT(*) AS n FROM jobs WHERE status IN ('queued','running') GROUP BY status")
    counts = {r["status"]: r["n"] for r in rows}
    return jsonify({"running": counts.get("running", 0), "queued": counts.get("queued", 0)})


@api.post("/jobs")
def submit_job():
    d = body()
    return jsonify(jobs().submit(d.get("kind") or "embedding", d.get("config") or {}))


@api.get("/jobs/<job_id>")
def get_job(job_id: str):
    job = jobs().get(job_id)
    if not job:
        raise KeyError(job_id)
    return jsonify(clean(job))


@api.post("/jobs/<job_id>/cancel")
def cancel_job(job_id: str):
    jobs().cancel(job_id)
    return jsonify({"ok": True})


@api.post("/jobs/<job_id>/rerun")
def rerun_job(job_id: str):
    return jsonify(jobs().rerun(job_id))


@api.delete("/jobs/<job_id>")
def delete_job(job_id: str):
    jobs().delete(job_id)
    return jsonify({"ok": True})


# ---- 結果 ---------------------------------------------------------------


@api.get("/results")
def results_api():
    rows = list_results(request.args.get("dataset") or None, request.args.get("kind") or None, ks_arg())
    return jsonify(rows)


@api.get("/results/datasets")
def results_datasets():
    rows = get_db().query("SELECT dataset_key, COUNT(*) AS n, MAX(created_at) AS latest FROM results GROUP BY dataset_key ORDER BY latest DESC")
    names = {d["key"]: d["name"] for d in datasets.list_datasets()}
    return jsonify([{**r, "name": names.get(r["dataset_key"], r["dataset_key"])} for r in rows])


@api.get("/results/<result_id>")
def result_api(result_id: str):
    detail = result_detail(result_id, ks_arg())
    if not detail:
        raise KeyError(result_id)
    return jsonify(detail)


@api.patch("/results/<result_id>")
def rename_result(result_id: str):
    label = (body().get("label") or "").strip()
    if not label:
        raise ValueError("標籤不能空白")
    if not get_result(result_id):
        raise KeyError(result_id)
    get_db().update("results", result_id, {"label": label})
    return jsonify({"ok": True})


@api.delete("/results/<result_id>")
def delete_result_api(result_id: str):
    delete_result(result_id)
    return jsonify({"ok": True})


@api.get("/results/<result_id>/curve")
def result_curve(result_id: str):
    return jsonify(k_curve(result_id, int(request.args.get("max_k", 20))))


@api.get("/results/<result_id>/queries")
def result_queries(result_id: str):
    a = request.args
    return jsonify(
        query_details(
            result_id,
            k=int(a.get("k", 5)),
            status=a.get("status", "all"),
            search=a.get("q", ""),
            page=int(a.get("page", 1)),
            per_page=min(int(a.get("per_page", 20)), 200),
            top_n=min(int(a.get("top_n", 5)), 20),
            sort=a.get("sort", "index"),
        )
    )


# ---- 匯出 ---------------------------------------------------------------


def _download(text: str, filename: str, mimetype: str) -> Response:
    return Response(
        text,
        mimetype=mimetype,
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@api.get("/export/leaderboard")
def export_leaderboard():
    ks = ks_arg()
    dataset = request.args.get("dataset") or None
    rows = list_results(dataset, request.args.get("kind") or None, ks)
    ids = [i for i in request.args.get("ids", "").split(",") if i]
    if ids:
        order = {i: n for n, i in enumerate(ids)}
        rows = sorted((r for r in rows if r["id"] in order), key=lambda r: order[r["id"]])
    fmt = request.args.get("format", "csv")
    metrics = tuple(m for m in request.args.get("metrics", "hit,mrr").split(",") if m in export.METRICS) or ("hit", "mrr")
    name = f"leaderboard-{dataset or 'all'}"
    if fmt == "json":
        return _download(export.leaderboard_json(rows, ks), f"{name}.json", "application/json")
    if fmt == "md":
        return _download(export.leaderboard_markdown(rows, ks, metrics), f"{name}.md", "text/markdown; charset=utf-8")
    return _download(export.leaderboard_csv(rows, ks), f"{name}.csv", "text/csv; charset=utf-8")


@api.get("/export/results/<result_id>")
def export_result(result_id: str):
    fmt = request.args.get("format", "json")
    if fmt == "queries_csv":
        k = int(request.args.get("k", 5))
        return _download(export.result_queries_csv(result_id, k), f"{result_id}-queries.csv", "text/csv; charset=utf-8")
    return _download(export.result_json(result_id, ks_arg()), f"{result_id}.json", "application/json")


# ---- 舊資料匯入 ---------------------------------------------------------


@api.post("/legacy/inspect")
def legacy_inspect():
    from pathlib import Path

    from embedbench.legacy import list_legacy_models

    path = (body().get("path") or "").strip()
    if not path or not Path(path).expanduser().exists():
        raise ValueError(f"找不到檔案：{path}")
    return jsonify(list_legacy_models(path))


@api.post("/legacy/import")
def legacy_import():
    return jsonify(jobs().submit("legacy_import", body()))
