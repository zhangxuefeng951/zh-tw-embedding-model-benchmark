(function () {
  "use strict";
  const { h, $, clear, api } = EB;

  const STATUS = {
    queued: ["排隊中", ""],
    running: ["執行中", "accent"],
    done: ["完成", "good"],
    failed: ["失敗", "bad"],
    cancelled: ["已取消", "warn"],
    interrupted: ["已中斷", "warn"],
  };
  const KIND = { embedding: "Embedding", bm25: "BM25", rerank: "Reranker", prepare_dataset: "資料集", legacy_import: "舊資料匯入" };
  const highlight = EB.params().get("highlight");
  const openErrors = new Set();

  function progressBar(job) {
    const p = job.progress || {};
    if (job.status === "queued") return h("div", { class: "small muted", text: "等待前面的任務完成…" });
    if (job.status !== "running") return null;
    const pct = p.total ? Math.min(100, (p.done / p.total) * 100) : 0;
    return h(
      "div",
      {},
      h("div", { class: "small secondary" }, p.stage || "準備中…", p.total ? ` — ${EB.fmtInt(p.done)} / ${EB.fmtInt(p.total)}（${pct.toFixed(0)}%）` : ""),
      h("div", { class: "progress" + (p.total ? "" : " indeterminate") }, h("div", { style: { width: pct + "%" } }))
    );
  }

  function action(label, fn, cls) {
    return h("button", { type: "button", class: "small " + (cls || ""), text: label, onclick: (e) => EB.busy(e.target, fn) });
  }

  function renderJob(job) {
    const [statusText, statusCls] = STATUS[job.status] || [job.status, ""];
    const actions = [];
    if (job.status === "queued" || job.status === "running")
      actions.push(action("取消", async () => {
        await api(`/api/jobs/${job.id}/cancel`, { method: "POST" });
        refresh();
      }, "danger"));
    else {
      actions.push(action("重新執行", async () => {
        await api(`/api/jobs/${job.id}/rerun`, { method: "POST" });
        EB.toast("已重新加入佇列（已算過的向量會沿用）");
        refresh();
      }));
      actions.push(action("刪除紀錄", async () => {
        await api(`/api/jobs/${job.id}`, { method: "DELETE" });
        refresh();
      }));
    }
    const results = job.result_ids || [];
    const messages = (job.progress && job.progress.messages) || [];
    const errorBox =
      job.status === "failed" || job.status === "interrupted" || job.status === "cancelled"
        ? h(
            "div",
            { class: "notice " + (job.status === "failed" ? "bad" : "warn") },
            h("div", { text: job.message || statusText }),
            job.error
              ? h(
                  "details",
                  {
                    open: openErrors.has(job.id),
                    ontoggle: (e) => (e.target.open ? openErrors.add(job.id) : openErrors.delete(job.id)),
                  },
                  h("summary", { class: "small", text: "錯誤詳細資訊" }),
                  h("pre", { class: "small", text: job.error })
                )
              : null
          )
        : null;
    return h(
      "div",
      { class: "job", id: "job-" + job.id, style: job.id === highlight ? { background: "var(--accent-wash)", margin: "0 -12px", padding: "14px 12px", borderRadius: "8px" } : {} },
      h(
        "div",
        { class: "row" },
        h("span", { class: "chip " + statusCls, text: statusText }),
        h("span", { class: "chip", text: KIND[job.kind] || job.kind }),
        h("span", { class: "job-title", text: job.title }),
        h("span", { class: "spacer" }),
        actions
      ),
      h(
        "div",
        { class: "small muted", style: { marginTop: "4px" } },
        `建立 ${EB.fmtTime(job.created_at)}`,
        job.started_at ? ` · 開始 ${EB.fmtTime(job.started_at)}` : "",
        job.started_at ? ` · 耗時 ${EB.duration(job.started_at, job.finished_at)}` : ""
      ),
      progressBar(job),
      messages.length ? h("div", { class: "notice warn" }, h("ul", {}, messages.map((m) => h("li", { text: m })))) : null,
      errorBox,
      results.length
        ? h(
            "div",
            { class: "row small", style: { marginTop: "6px" } },
            `產生 ${results.length} 個結果：`,
            results.slice(0, 12).map((id) => h("a", { href: `/results/${id}`, text: id.slice(-6) })),
            results.length > 12 ? "…" : null,
            h("a", { href: "/", text: "→ 排行榜" })
          )
        : null
    );
  }

  let timer = null;
  async function refresh() {
    clearTimeout(timer);
    try {
      const jobs = await api("/api/jobs");
      const box = clear($("#jobs"));
      if (!jobs.length) box.appendChild(h("div", { class: "empty", text: "還沒有任務。" }));
      jobs.forEach((j) => box.appendChild(renderJob(j)));
      const active = jobs.some((j) => j.status === "running" || j.status === "queued");
      timer = setTimeout(refresh, active ? 1500 : 5000);
    } catch (e) {
      timer = setTimeout(refresh, 5000);
    }
  }

  document.addEventListener(
    "DOMContentLoaded",
    EB.guard(async () => {
      const s = await api("/api/settings");
      $("#max-jobs").textContent = s.max_concurrent_jobs;
      await refresh();
      if (highlight) {
        const el = document.getElementById("job-" + highlight);
        if (el) el.scrollIntoView({ block: "center" });
      }
    })
  );
})();
