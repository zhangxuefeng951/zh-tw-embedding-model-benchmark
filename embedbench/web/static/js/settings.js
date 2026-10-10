(function () {
  "use strict";
  const { h, $, clear, api } = EB;

  async function loadSettings() {
    const s = await api("/api/settings");
    const form = $("#settings-form");
    form.max_concurrent_jobs.value = s.max_concurrent_jobs;
    form.default_ks.value = s.default_ks.join(",");
    $("#data-dir").textContent = s.data_dir;
    $("#presets-dir").textContent = s.presets_dir;
  }

  async function loadEnv() {
    const env = await api("/api/env");
    const table = clear($("#env"));
    table.appendChild(
      h(
        "tbody",
        {},
        Object.entries(env).map(([k, ok]) => h("tr", {}, h("td", { class: "mono", text: k }), h("td", {}, h("span", { class: "chip " + (ok ? "good" : ""), text: ok ? "已設定" : "未設定" }))))
      )
    );
  }

  async function inspectLegacy() {
    const path = $("#legacy-path").value.trim();
    const models = await api("/api/legacy/inspect", { method: "POST", body: { path } });
    const box = clear($("#legacy-models"));
    if (!models.length) {
      box.appendChild(h("div", { class: "notice warn", text: "這個資料庫沒有任何模型資料。" }));
      return;
    }
    const checks = models.map((m) => h("input", { type: "checkbox", value: m.model, checked: true }));
    box.appendChild(h("p", { class: "small secondary", style: { marginTop: "12px" }, text: `找到 ${models.length} 個模型，勾選要匯入的模型（同一個模型重複匯入會取代上次的結果）：` }));
    box.appendChild(
      h(
        "div",
        { class: "stack" },
        models.map((m, i) => h("label", { class: "check" }, checks[i], h("span", { class: "mono", text: m.model }), h("span", { class: "muted small", text: `${m.questions} 題 / ${m.paragraphs} 段落` })))
      )
    );
    box.appendChild(
      h(
        "div",
        { class: "row", style: { marginTop: "12px" } },
        h("button", {
          type: "button",
          class: "primary",
          text: "匯入",
          onclick: (e) =>
            EB.busy(e.target, async () => {
              const selected = checks.filter((c) => c.checked).map((c) => c.value);
              if (!selected.length) throw new Error("請至少勾選一個模型");
              const job = await api("/api/legacy/import", { method: "POST", body: { path, dataset: $("#legacy-dataset").value, models: selected } });
              location.href = "/jobs?highlight=" + job.id;
            }),
        })
      )
    );
  }

  document.addEventListener(
    "DOMContentLoaded",
    EB.guard(async () => {
      $("#settings-form").addEventListener("submit", (e) => {
        e.preventDefault();
        const form = e.target;
        EB.busy(form.querySelector("button"), async () => {
          await api("/api/settings", { method: "PUT", body: { max_concurrent_jobs: parseInt(form.max_concurrent_jobs.value, 10), default_ks: form.default_ks.value } });
          EB.toast("已儲存設定");
          await loadSettings();
        });
      });
      $("#legacy-inspect").addEventListener("click", (e) => EB.busy(e.target, inspectLegacy));
      $("#clear-cache").addEventListener("click", (e) =>
        EB.busy(e.target, async () => {
          if (!confirm("清除所有向量 / reranker / BM25 快取？已完成的評測結果不受影響，但之後重跑需要重新計算。")) return;
          const out = await api("/api/cache/clear", { method: "POST" });
          EB.toast(`已清除：向量 ${out.vectors} 組、reranker ${out.rerank} 組、BM25 ${out.bm25} 組`);
        })
      );
      await Promise.all([loadSettings(), loadEnv()]);
    })
  );
})();
