"""背景任務佇列：同時執行的任務數由設定 max_concurrent_jobs 決定（預設 1）。"""

from __future__ import annotations

import threading
import time
import traceback

from embedbench.db import Database, get_db, get_settings, now_iso
from embedbench.runner import NORMALIZERS, RUNNERS, Cancelled, JobContext, job_title, new_id

ACTIVE = ("queued", "running")


class JobManager:
    def __init__(self, db: Database | None = None):
        self.db = db or get_db()
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._threads: dict[str, threading.Thread] = {}
        self._cancel: dict[str, threading.Event] = {}
        self._progress: dict[str, dict] = {}
        self._last_write: dict[str, float] = {}
        self._scheduler: threading.Thread | None = None

    # ---- 生命週期 -------------------------------------------------------

    def start(self) -> None:
        if self._scheduler and self._scheduler.is_alive():
            return
        # 上次程式結束時還在跑的任務：標記為中斷（重新執行會從向量快取接續）
        self.db.execute(
            "UPDATE jobs SET status = 'interrupted', finished_at = ?, message = ? WHERE status = 'running'",
            (now_iso(), "程式重新啟動，任務被中斷；按「重新執行」會從已完成的部分繼續"),
        )
        self._scheduler = threading.Thread(target=self._loop, name="job-scheduler", daemon=True)
        self._scheduler.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        for ev in self._cancel.values():
            ev.set()

    # ---- 操作 -----------------------------------------------------------

    def submit(self, kind: str, config: dict) -> dict:
        if kind not in RUNNERS:
            raise ValueError(f"未知的任務類型：{kind}")
        normalized = NORMALIZERS[kind](config)
        job = {
            "id": new_id("j"),
            "kind": kind,
            "status": "queued",
            "title": job_title(kind, normalized),
            "config": normalized,
            "progress": {"stage": "排隊中", "done": 0, "total": 0},
            "created_at": now_iso(),
        }
        self.db.insert("jobs", job)
        self._wake.set()
        return job

    def cancel(self, job_id: str) -> None:
        job = self.get(job_id)
        if not job:
            raise KeyError(job_id)
        if job["status"] == "queued":
            self.db.update("jobs", job_id, {"status": "cancelled", "finished_at": now_iso(), "message": "已取消"})
        elif job["status"] == "running":
            ev = self._cancel.get(job_id)
            if ev:
                ev.set()

    def rerun(self, job_id: str) -> dict:
        job = self.get(job_id)
        if not job:
            raise KeyError(job_id)
        return self.submit(job["kind"], job["config"])

    def delete(self, job_id: str) -> None:
        job = self.get(job_id)
        if job and job["status"] in ACTIVE:
            raise ValueError("執行中或排隊中的任務不能刪除，請先取消")
        self.db.execute("DELETE FROM jobs WHERE id = ?", (job_id,))

    def get(self, job_id: str) -> dict | None:
        job = self.db.query_one("SELECT * FROM jobs WHERE id = ?", (job_id,), table="jobs")
        if job and job_id in self._progress and job["status"] == "running":
            job["progress"] = self._progress[job_id]
        return job

    def list(self, limit: int = 100) -> list[dict]:
        jobs = self.db.query("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,), table="jobs")
        for job in jobs:
            if job["status"] == "running" and job["id"] in self._progress:
                job["progress"] = self._progress[job["id"]]
        return jobs

    # ---- 排程 -----------------------------------------------------------

    def _running_count(self) -> int:
        self._threads = {k: t for k, t in self._threads.items() if t.is_alive()}
        return len(self._threads)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(timeout=1.0)
            self._wake.clear()
            with self._lock:
                limit = max(1, int(get_settings(self.db)["max_concurrent_jobs"]))
                free = limit - self._running_count()
                if free <= 0:
                    continue
                queued = self.db.query(
                    "SELECT * FROM jobs WHERE status = 'queued' ORDER BY created_at LIMIT ?", (free,), table="jobs"
                )
                for job in queued:
                    self._start(job)

    def _start(self, job: dict) -> None:
        cancel = threading.Event()
        self._cancel[job["id"]] = cancel
        self.db.update("jobs", job["id"], {"status": "running", "started_at": now_iso()})
        t = threading.Thread(target=self._run, args=(job, cancel), name=f"job-{job['id']}", daemon=True)
        self._threads[job["id"]] = t
        t.start()

    def _report(self, job_id: str, progress: dict) -> None:
        self._progress[job_id] = dict(progress)
        now = time.monotonic()
        if now - self._last_write.get(job_id, 0) > 1.0:
            self._last_write[job_id] = now
            self.db.update("jobs", job_id, {"progress": progress})

    def _run(self, job: dict, cancel: threading.Event) -> None:
        job_id = job["id"]
        ctx = JobContext(job_id, cancel, lambda p: self._report(job_id, p))
        values: dict = {"finished_at": None}
        try:
            result_ids = RUNNERS[job["kind"]](job["config"], ctx)
            values = {"status": "done", "result_ids": result_ids, "message": "完成"}
        except Cancelled:
            values = {"status": "cancelled", "message": "已取消（已完成的向量已存入快取，重新執行會接續）"}
        except Exception as e:
            values = {"status": "failed", "message": str(e) or e.__class__.__name__, "error": traceback.format_exc()}
        finally:
            progress = self._progress.pop(job_id, None) or {}
            progress["messages"] = ctx.messages
            values.update({"finished_at": now_iso(), "progress": progress})
            self.db.update("jobs", job_id, values)
            self._cancel.pop(job_id, None)
            self._last_write.pop(job_id, None)
            self._wake.set()


def run_job_sync(kind: str, config: dict, on_progress=None, db: Database | None = None) -> dict:
    """CLI 使用：在目前的執行緒直接執行任務，並記錄到任務表（Web UI 看得到）。"""
    db = db or get_db()
    normalized = NORMALIZERS[kind](config)
    job_id = new_id("j")
    db.insert(
        "jobs",
        {
            "id": job_id,
            "kind": kind,
            "status": "running",
            "title": job_title(kind, normalized) + "（CLI）",
            "config": normalized,
            "created_at": now_iso(),
            "started_at": now_iso(),
        },
    )
    ctx = JobContext(job_id, threading.Event(), on_progress)
    try:
        result_ids = RUNNERS[kind](normalized, ctx)
    except BaseException as e:
        status = "cancelled" if isinstance(e, (Cancelled, KeyboardInterrupt)) else "failed"
        db.update(
            "jobs",
            job_id,
            {
                "status": status,
                "message": str(e) or e.__class__.__name__,
                "error": traceback.format_exc(),
                "finished_at": now_iso(),
                "progress": {"messages": ctx.messages},
            },
        )
        raise
    db.update(
        "jobs",
        job_id,
        {
            "status": "done",
            "result_ids": result_ids,
            "message": "完成",
            "finished_at": now_iso(),
            "progress": {"messages": ctx.messages},
        },
    )
    return {"id": job_id, "result_ids": result_ids, "messages": ctx.messages}
