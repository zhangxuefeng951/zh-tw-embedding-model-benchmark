"""Flask Web UI。"""

from __future__ import annotations

from flask import Flask, render_template

from embedbench import __version__
from embedbench.jobs import JobManager

PAGES = [
    ("leaderboard", "/", "排行榜"),
    ("new_run", "/new", "新增評測"),
    ("jobs", "/jobs", "任務"),
    ("datasets", "/datasets", "資料集"),
    ("presets", "/presets", "預設檔"),
    ("settings", "/settings", "設定"),
]


def create_app(start_jobs: bool = True) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024  # 上傳資料集最大 1 GB
    app.config["JSON_AS_ASCII"] = False
    app.json.ensure_ascii = False
    app.json.sort_keys = False

    jobs = JobManager()
    app.extensions["jobs"] = jobs
    if start_jobs:
        jobs.start()

    from embedbench.web.api import api

    app.register_blueprint(api)

    def page(name: str, template: str):
        def view(**kwargs):
            return render_template(template, page=name, pages=PAGES, version=__version__, **kwargs)

        view.__name__ = f"page_{name}"
        return view

    for name, path, _ in PAGES:
        app.add_url_rule(path, view_func=page(name, f"{name}.html"))
    app.add_url_rule("/results/<result_id>", view_func=page("result", "result.html"))
    app.add_url_rule("/results/<result_id>/queries", view_func=page("queries", "queries.html"))
    return app
