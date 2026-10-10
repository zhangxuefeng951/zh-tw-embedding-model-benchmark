(function () {
  "use strict";
  const { h, $, clear, api, fmt } = EB;

  const METRICS = [
    ["hit", "Hit"],
    ["mrr", "MRR"],
    ["ndcg", "NDCG"],
    ["recall", "Recall"],
  ];
  const KINDS = [
    ["embedding", "Embedding"],
    ["rerank", "Reranker"],
    ["bm25", "BM25"],
    ["legacy", "舊資料"],
  ];

  const state = {
    datasets: [],
    dataset: null,
    rows: [],
    ks: [1, 3, 5, 10],
    metrics: new Set(["hit", "mrr"]),
    kinds: new Set(KINDS.map((k) => k[0])),
    search: "",
    showTrials: true,
    sort: null,
    asc: false,
    selected: new Set(),
    chartMetric: null,
  };

  function load(key, fallback) {
    try {
      const v = localStorage.getItem("eb-lb-" + key);
      return v ? JSON.parse(v) : fallback;
    } catch {
      return fallback;
    }
  }
  function save(key, value) {
    try {
      localStorage.setItem("eb-lb-" + key, JSON.stringify(value));
    } catch {
      /* 無法儲存時忽略 */
    }
  }

  function metricCols() {
    const cols = [];
    for (const [m] of METRICS) if (state.metrics.has(m)) for (const k of state.ks) cols.push(`${m}@${k}`);
    return cols;
  }

  function visibleRows() {
    const q = state.search.trim().toLowerCase();
    let rows = state.rows.filter((r) => state.kinds.has(r.kind) && (state.showTrials || !r.query_limit));
    if (q) {
      rows = rows.filter((r) =>
        [r.label, r.model, r.provider, r.pair_label, r.query_prefix, r.document_prefix, String(r.dims || "")]
          .filter(Boolean)
          .some((s) => s.toLowerCase().includes(q))
      );
    }
    const key = state.sort;
    if (key) {
      rows = rows.slice().sort((a, b) => {
        const va = key in a ? a[key] : a.metrics[key];
        const vb = key in b ? b[key] : b.metrics[key];
        const cmp = typeof va === "string" ? String(va).localeCompare(String(vb), "zh-Hant") : (va ?? -Infinity) - (vb ?? -Infinity);
        return state.asc ? cmp : -cmp;
      });
    }
    return rows;
  }

  function defaultSort() {
    const cols = metricCols();
    return cols.includes("hit@5") ? "hit@5" : cols[0] || "created_at";
  }

  function renderToggles() {
    const mt = clear($("#metric-toggles"));
    for (const [m, label] of METRICS) {
      mt.appendChild(
        h("span", {
          class: "chip toggle" + (state.metrics.has(m) ? " on" : ""),
          text: label,
          role: "button",
          tabindex: "0",
          onclick: () => {
            if (state.metrics.has(m) && state.metrics.size > 1) state.metrics.delete(m);
            else state.metrics.add(m);
            save("metrics", [...state.metrics]);
            renderToggles();
            render();
          },
        })
      );
    }
    const kt = clear($("#kind-toggles"));
    for (const [k, label] of KINDS) {
      const n = state.rows.filter((r) => r.kind === k).length;
      if (!n) continue;
      kt.appendChild(
        h("span", {
          class: "chip toggle" + (state.kinds.has(k) ? " on" : ""),
          text: `${label} ${n}`,
          role: "button",
          tabindex: "0",
          onclick: () => {
            if (state.kinds.has(k)) state.kinds.delete(k);
            else state.kinds.add(k);
            renderToggles();
            render();
          },
        })
      );
    }
    const trials = state.rows.filter((r) => r.query_limit).length;
    if (trials)
      kt.appendChild(
        h("span", {
          class: "chip toggle" + (state.showTrials ? " on" : ""),
          text: `快速試跑 ${trials}`,
          title: "只跑前 N 題的結果；題數不同的結果不宜直接比較",
          role: "button",
          tabindex: "0",
          onclick: () => {
            state.showTrials = !state.showTrials;
            renderToggles();
            render();
          },
        })
      );
  }

  function renderChartSelect() {
    const sel = clear($("#chart-metric"));
    const cols = metricCols();
    if (!cols.includes(state.chartMetric)) state.chartMetric = cols.includes("hit@5") ? "hit@5" : cols[0];
    for (const c of cols) sel.appendChild(h("option", { value: c, text: EB.metricLabel(c), selected: c === state.chartMetric }));
  }

  function renderChart(rows) {
    const metric = state.chartMetric;
    let items = rows.filter((r) => state.selected.has(r.id));
    const picked = items.length > 0;
    if (!picked) items = rows.slice();
    items.sort((a, b) => (b.metrics[metric] ?? -1) - (a.metrics[metric] ?? -1));
    const limit = picked ? items.length : 15;
    const shown = items.slice(0, limit);
    $("#chart-title").textContent = `${EB.metricLabel(metric)} 比較`;
    $("#chart-note").textContent = picked
      ? `比較勾選的 ${shown.length} 個結果。`
      : `顯示前 ${shown.length} 名（共 ${items.length} 個）；在下方表格勾選可以只比較特定結果。灰色長條是 BM25 基準。`;
    EB.barChart(
      $("#chart"),
      shown.map((r) => ({
        label: r.label,
        title: `${r.label}｜${r.pair_label || ""}`,
        sub: [r.pair_label, r.dims ? `${r.dims} 維` : null, r.query_limit ? `試跑 ${r.num_queries} 題` : null].filter(Boolean).join(" · "),
        value: r.metrics[metric],
        muted: r.kind === "bm25",
        onclick: () => (location.href = `/results/${r.id}`),
      })),
      { metricLabel: EB.metricLabel(metric), emptyText: "沒有符合條件的結果" }
    );
  }

  function prefixTitle(r) {
    if (r.query_prefix === null || r.query_prefix === undefined) return "";
    const show = (s) => (s ? JSON.stringify(s) : "（無）");
    return `查詢前綴：${show(r.query_prefix)}\n文件前綴：${show(r.document_prefix)}`;
  }

  function renderTable(rows) {
    const table = clear($("#table"));
    const cols = metricCols();
    const best = {};
    for (const c of cols) best[c] = Math.max(...rows.map((r) => r.metrics[c] ?? -1));

    const sortable = (key, label, cls) => {
      const th = h("th", { class: "sortable " + (cls || "") + (state.sort === key ? " sorted" + (state.asc ? " asc" : "") : ""), text: label });
      th.addEventListener("click", () => {
        if (state.sort === key) state.asc = !state.asc;
        else {
          state.sort = key;
          state.asc = typeof (rows[0] || {})[key] === "string";
        }
        render();
      });
      return th;
    };
    const allChecked = rows.length > 0 && rows.every((r) => state.selected.has(r.id));
    const headCheck = h("input", {
      type: "checkbox",
      checked: allChecked,
      "aria-label": "全選",
      onchange: (e) => {
        rows.forEach((r) => (e.target.checked ? state.selected.add(r.id) : state.selected.delete(r.id)));
        render();
      },
    });
    table.appendChild(
      h(
        "thead",
        {},
        h(
          "tr",
          {},
          h("th", {}, headCheck),
          h("th", { class: "num", text: "#" }),
          sortable("label", "模型 / 標籤"),
          h("th", { text: "前綴組合" }),
          sortable("dims", "維度", "num"),
          sortable("num_queries", "題數", "num"),
          cols.map((c) => sortable(c, EB.metricLabel(c), "num")),
          sortable("created_at", "時間")
        )
      )
    );
    const tbody = h("tbody");
    rows.forEach((r, i) => {
      const chips = [h("span", { class: "chip" + (r.kind === "embedding" ? "" : " accent"), text: r.kind_label })];
      if (r.query_limit) chips.push(h("span", { class: "chip warn", text: `試跑 ${r.num_queries} 題` }));
      if (r.truncated) chips.push(h("span", { class: "chip", text: "截取維度" }));
      const cut = r.truncated_texts ? (r.truncated_texts.document || 0) + (r.truncated_texts.query || 0) : 0;
      if (cut) chips.push(h("span", { class: "chip warn", title: "超過模型輸入長度上限、自動從尾端截短的文字筆數", text: `截短 ${cut} 筆` }));
      tbody.appendChild(
        h(
          "tr",
          {},
          h(
            "td",
            {},
            h("input", {
              type: "checkbox",
              checked: state.selected.has(r.id),
              "aria-label": "勾選 " + r.label,
              onchange: (e) => {
                e.target.checked ? state.selected.add(r.id) : state.selected.delete(r.id);
                render();
              },
            })
          ),
          h("td", { class: "num muted", text: i + 1 }),
          h(
            "td",
            { style: { minWidth: "220px" } },
            h("a", { href: `/results/${r.id}`, text: r.label }),
            h("span", { class: "sub" }, [r.model, r.provider].filter(Boolean).join(" · ")),
            h("div", { class: "row", style: { gap: "4px", marginTop: "2px" } }, chips)
          ),
          h("td", { title: prefixTitle(r), style: { minWidth: "170px", maxWidth: "260px" }, text: r.pair_label || "" }),
          h("td", { class: "num", text: r.dims ?? "—" }),
          h("td", { class: "num", text: EB.fmtInt(r.num_queries) }),
          cols.map((c) => h("td", { class: "num" + (r.metrics[c] === best[c] && rows.length > 1 ? " best" : ""), text: fmt(r.metrics[c]) })),
          h("td", { class: "small muted nowrap", text: EB.fmtTime(r.created_at) })
        )
      );
    });
    if (!rows.length) tbody.appendChild(h("tr", {}, h("td", { colspan: 7 + cols.length, class: "empty", text: "沒有符合條件的結果" })));
    table.appendChild(tbody);
  }

  function updateExport(rows) {
    const ids = (state.selected.size ? rows.filter((r) => state.selected.has(r.id)) : rows).map((r) => r.id);
    const base = new URLSearchParams({ dataset: state.dataset, ks: state.ks.join(","), metrics: [...state.metrics].join(","), ids: ids.join(",") });
    for (const fmtName of ["csv", "json", "md"]) {
      base.set("format", fmtName);
      $("#export-" + fmtName).href = "/api/export/leaderboard?" + base.toString();
    }
    $("#selected-count").textContent = state.selected.size ? `已勾選 ${state.selected.size} 個` : "";
    $("#clear-selection").classList.toggle("hidden", !state.selected.size);
    $("#delete-selected").classList.toggle("hidden", !state.selected.size);
  }

  function render() {
    renderChartSelect();
    const rows = visibleRows();
    $("#count").textContent = `${rows.length} 個`;
    renderChart(rows);
    renderTable(rows);
    updateExport(rows);
  }

  async function loadRows() {
    state.rows = await api(`/api/results?dataset=${encodeURIComponent(state.dataset)}&ks=${state.ks.join(",")}`);
    const ids = new Set(state.rows.map((r) => r.id));
    state.selected = new Set([...state.selected].filter((id) => ids.has(id)));
    if (!state.sort) state.sort = defaultSort();
    renderToggles();
    render();
  }

  async function init() {
    const settings = await api("/api/settings");
    state.ks = load("ks", settings.default_ks);
    state.metrics = new Set(load("metrics", ["hit", "mrr"]));
    $("#ks").value = state.ks.join(",");
    state.datasets = await api("/api/results/datasets");
    if (!state.datasets.length) {
      $("#empty").classList.remove("hidden");
      return;
    }
    $("#board").classList.remove("hidden");
    const wanted = EB.params().get("dataset") || load("dataset", null);
    state.dataset = state.datasets.some((d) => d.dataset_key === wanted) ? wanted : state.datasets[0].dataset_key;
    const sel = $("#dataset");
    for (const d of state.datasets) sel.appendChild(h("option", { value: d.dataset_key, text: `${d.name}（${d.n} 個結果）`, selected: d.dataset_key === state.dataset }));
    sel.addEventListener(
      "change",
      EB.guard(async () => {
        state.dataset = sel.value;
        save("dataset", state.dataset);
        history.replaceState(null, "", "?dataset=" + encodeURIComponent(state.dataset));
        state.selected.clear();
        await loadRows();
      })
    );
    $("#ks").addEventListener(
      "change",
      EB.guard(async () => {
        const ks = EB.parseKs($("#ks").value);
        if (!ks.length) throw new Error("K 值格式錯誤，例如 1,3,5,10");
        state.ks = ks;
        save("ks", ks);
        $("#ks").value = ks.join(",");
        if (!metricCols().includes(state.sort)) state.sort = defaultSort();
        await loadRows();
      })
    );
    $("#search").addEventListener("input", () => {
      state.search = $("#search").value;
      render();
    });
    $("#chart-metric").addEventListener("change", () => {
      state.chartMetric = $("#chart-metric").value;
      render();
    });
    $("#clear-selection").addEventListener("click", () => {
      state.selected.clear();
      render();
    });
    $("#delete-selected").addEventListener(
      "click",
      EB.guard(async () => {
        const n = state.selected.size;
        if (!n || !confirm(`確定刪除勾選的 ${n} 個結果？（向量快取不受影響）`)) return;
        for (const id of state.selected) await api(`/api/results/${id}`, { method: "DELETE" });
        state.selected.clear();
        EB.toast(`已刪除 ${n} 個結果`);
        await loadRows();
      })
    );
    document.addEventListener("eb-theme", render);
    await loadRows();
  }

  document.addEventListener("DOMContentLoaded", EB.guard(init));
})();
