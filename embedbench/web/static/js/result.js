(function () {
  "use strict";
  const { h, $, clear, api, fmt } = EB;
  const root = $("#result");
  const id = root.dataset.id;

  const METRICS = [
    ["hit", "Hit@K"],
    ["mrr", "MRR@K"],
    ["ndcg", "NDCG@K"],
    ["recall", "Recall@K"],
  ];
  const SECRET_HINT = /key|token|secret|password/i;

  function kv(pairs) {
    return h(
      "dl",
      { class: "kv" },
      pairs.filter(Boolean).flatMap(([k, v]) => [h("dt", { text: k }), h("dd", {}, v instanceof Node ? v : String(v ?? "—"))])
    );
  }

  function metricsTable(r, ks) {
    const all = r.all_metrics || {};
    return h(
      "table",
      {},
      h("thead", {}, h("tr", {}, h("th", { text: "K" }), METRICS.map(([, l]) => h("th", { class: "num", text: l })))),
      h(
        "tbody",
        {},
        ks.map((k) => h("tr", {}, h("td", { text: k }), METRICS.map(([m]) => h("td", { class: "num", text: fmt(all[`${m}@${k}`]) }))))
      )
    );
  }

  function healthTable(health) {
    if (!health) return h("p", { class: "muted", text: "沒有向量資訊（BM25 / Reranker 結果）。" });
    const sides = Object.entries(health);
    const fields = [
      ["count", "筆數"],
      ["dims", "維度"],
      ["nan_rows", "含 NaN"],
      ["inf_rows", "含無限大"],
      ["zero_rows", "全零向量"],
      ["norm_min", "長度最小"],
      ["norm_mean", "長度平均"],
      ["norm_max", "長度最大"],
    ];
    const bad = sides.some(([, v]) => v.nan_rows || v.inf_rows || v.zero_rows);
    return h(
      "div",
      {},
      bad ? h("div", { class: "notice bad", text: "有異常向量（NaN / 無限大 / 全零），結果可能不可信。若是 EmbeddingGemma，請確認沒有使用 float16。" }) : h("div", { class: "notice good", text: "沒有 NaN、無限大或全零向量。" }),
      h(
        "table",
        {},
        h("thead", {}, h("tr", {}, h("th", { text: "" }), sides.map(([s]) => h("th", { class: "num", text: s === "query" ? "查詢" : "文件" })))),
        h(
          "tbody",
          {},
          fields.map(([f, label]) =>
            h(
              "tr",
              {},
              h("td", { text: label }),
              sides.map(([, v]) => h("td", { class: "num", text: f.startsWith("norm_") && typeof v[f] === "number" ? v[f].toFixed(4) : v[f] ?? "—" }))
            )
          )
        )
      ),
      h("p", { class: "small muted", style: { marginTop: "6px" }, text: "評測時一律先正規化成長度 1 再算 cosine，所以原始長度不影響分數；長度接近 1 代表服務端已經正規化。" })
    );
  }

  function paramsTable(params) {
    const entries = Object.entries(params || {}).filter(([, v]) => v !== null && v !== "" && !(typeof v === "object" && !Object.keys(v).length));
    return h(
      "table",
      {},
      h(
        "tbody",
        {},
        entries.map(([k, v]) =>
          h("tr", {}, h("td", { class: "mono", text: k }), h("td", { class: "mono", text: SECRET_HINT.test(k) && !k.endsWith("_env") ? "••••" : typeof v === "object" ? JSON.stringify(v) : String(v) }))
        )
      )
    );
  }

  function timingText(t) {
    if (!t) return "—";
    const parts = [];
    if (t.document_embed) parts.push(`文件向量 ${t.document_embed.seconds} 秒（沿用快取 ${t.document_embed.cached}/${t.document_embed.total}）`);
    if (t.query_embed) parts.push(`查詢向量 ${t.query_embed.seconds} 秒（沿用快取 ${t.query_embed.cached}/${t.query_embed.total}）`);
    if (t.rank_seconds !== undefined) parts.push(`檢索評分 ${t.rank_seconds} 秒`);
    if (t.seconds !== undefined) parts.push(`${t.seconds} 秒`);
    return parts.join("；");
  }

  async function render() {
    const settings = await api("/api/settings");
    const ks = Array.from(new Set([...settings.default_ks, 1, 3, 5, 10, 20, 50, 100])).sort((a, b) => a - b);
    const r = await api(`/api/results/${id}?ks=${ks.join(",")}`);
    const datasets = await api("/api/datasets");
    const ds = datasets.find((d) => d.key === r.dataset_key);
    document.title = `${r.label} · 評測結果`;
    const cfg = r.config || {};
    clear(root);

    // ---- 標題
    const chips = [
      h("span", { class: "chip accent", text: r.kind_label }),
      h("a", { class: "chip", href: `/?dataset=${encodeURIComponent(r.dataset_key)}`, text: ds ? ds.name : r.dataset_key }),
      r.dims ? h("span", { class: "chip", text: `${r.dims} 維${r.truncated ? `（由 ${cfg.full_dims} 維截取）` : ""}` }) : null,
      h("span", { class: "chip", text: `${EB.fmtInt(r.num_queries)} 題 / ${EB.fmtInt(r.num_documents)} 文件` }),
      r.query_limit ? h("span", { class: "chip warn", text: `快速試跑：前 ${r.query_limit} 題` }) : null,
    ];
    root.appendChild(
      h(
        "div",
        { class: "page-head" },
        h(
          "div",
          {},
          h(
            "h1",
            {},
            r.label,
            " ",
            h("button", {
              type: "button",
              class: "small",
              text: "改名",
              onclick: EB.guard(async () => {
                const label = await EB.promptDialog("修改顯示名稱", "名稱", r.label);
                if (!label) return;
                await api(`/api/results/${id}`, { method: "PATCH", body: { label } });
                render();
              }),
            })
          ),
          h("div", { class: "row", style: { gap: "6px" } }, chips),
          h("p", { class: "small muted", style: { marginTop: "6px" } }, [r.model, r.provider, `建立於 ${EB.fmtTime(r.created_at)}`, `結果 ID ${r.id}`].filter(Boolean).join(" · "))
        ),
        h(
          "div",
          { class: "row" },
          h("a", { class: "btn primary", href: `/results/${id}/queries`, text: "逐題錯誤分析" }),
          h("a", { class: "btn", href: `/new?tab=rerank&source=${id}`, text: "用這個結果做 Reranker" }),
          h("a", { class: "btn", href: `/api/export/results/${id}?format=json&ks=${ks.join(",")}`, text: "匯出 JSON" }),
          h("a", { class: "btn", href: `/api/export/results/${id}?format=queries_csv&k=5`, text: "匯出逐題 CSV" }),
          h("button", {
            type: "button",
            class: "danger",
            text: "刪除",
            onclick: EB.guard(async () => {
              if (!confirm("確定刪除這個結果？")) return;
              await api(`/api/results/${id}`, { method: "DELETE" });
              location.href = `/?dataset=${encodeURIComponent(r.dataset_key)}`;
            }),
          })
        )
      )
    );

    if (cfg.notes && cfg.notes.length) root.appendChild(h("div", { class: "notice warn" }, h("ul", {}, cfg.notes.map((n) => h("li", { text: n })))));
    if (r.kind === "legacy") root.appendChild(h("div", { class: "notice", text: cfg.note || "舊資料匯入：前綴等設定未記錄" }));

    // ---- 指標 + 曲線
    const chartBox = h("div", {});
    root.appendChild(
      h(
        "div",
        { class: "grid-2" },
        h("div", { class: "card" }, h("h2", { text: "指標" }), h("div", { class: "table-wrap" }, metricsTable(r, ks)), h("p", { class: "small muted", style: { marginTop: "8px" }, text: "每題只有一個正解時，Recall@K 等於 Hit@K。" })),
        h("div", { class: "card" }, h("h2", { text: "K 值曲線" }), chartBox)
      )
    );
    const curve = await api(`/api/results/${id}/curve?max_k=20`);
    const sameRecall = curve.recall.every((v, i) => Math.abs(v - curve.hit[i]) < 1e-9);
    const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
    const drawCurve = () => {
      const series = [
        { name: sameRecall ? "Hit（= Recall）" : "Hit", values: curve.hit, color: css("--series-1") },
        { name: "NDCG", values: curve.ndcg, color: css("--series-2") },
        { name: "MRR", values: curve.mrr, color: css("--series-3") },
      ];
      if (!sameRecall) series.push({ name: "Recall", values: curve.recall, color: css("--series-4") });
      EB.lineChart(chartBox, { xs: curve.ks, series }, { xticks: [1, 5, 10, 15, 20], xName: "K", ariaLabel: "Hit、NDCG、MRR 隨 K 變化" });
    };
    redraw = drawCurve;
    drawCurve();

    // ---- 設定
    const settingsCard = h("div", { class: "card" }, h("h2", { text: "設定" }));
    if (r.kind === "embedding" || r.kind === "rerank") {
      settingsCard.appendChild(
        kv([
          ["前綴組合", r.pair_label],
          ["查詢前綴", h("span", { class: "mono" }, EB.prefixNode(r.query_prefix))],
          ["文件前綴", h("span", { class: "mono" }, EB.prefixNode(r.document_prefix))],
          r.kind === "rerank" ? ["Reranker", cfg.reranker] : ["Provider", cfg.provider],
          r.kind === "rerank" ? ["候選數量", cfg.candidates] : null,
          r.kind === "rerank" ? ["文件加標題", cfg.include_title ? "是" : "否"] : null,
          cfg.preset_id ? ["預設檔", cfg.preset_id] : null,
        ])
      );
      settingsCard.appendChild(h("h3", { text: "參數" }));
      settingsCard.appendChild(h("div", { class: "table-wrap" }, paramsTable(cfg.params)));
    } else if (r.kind === "bm25") {
      settingsCard.appendChild(kv([["斷詞", "jieba"], ["k1", cfg.k1], ["b", cfg.b]]));
    } else {
      settingsCard.appendChild(kv([["來源", cfg.source_db], ["舊模型名稱", cfg.legacy_model], ["未對應的段落", cfg.unmatched_paragraphs], ["未對應的問題", cfg.unmatched_questions]]));
    }
    settingsCard.appendChild(h("h3", { text: "耗時" }));
    settingsCard.appendChild(h("p", { class: "small secondary", text: timingText(r.timing) }));
    if (r.job_id) settingsCard.appendChild(h("p", { class: "small muted" }, "任務 ", h("a", { href: `/jobs?highlight=${r.job_id}`, text: r.job_id })));

    const healthCard = h("div", { class: "card" }, h("h2", { text: "向量檢查" }), healthTable(r.health));
    root.appendChild(h("div", { class: "grid-2" }, settingsCard, healthCard));

    // ---- 關聯結果
    if (r.parent || (r.children && r.children.length)) {
      const card = h("div", { class: "card" }, h("h2", { text: "關聯結果" }));
      if (r.parent)
        card.appendChild(
          h("p", {}, "第一階段：", h("a", { href: `/results/${r.parent.id}`, text: r.parent.label }), ` · ${r.parent.pair_label || ""} · Hit@5 ${fmt(r.parent.metrics["hit@5"])} → ${fmt(r.metrics["hit@5"])}`)
        );
      if (r.children && r.children.length) {
        card.appendChild(h("p", { class: "small secondary", text: "以這個結果為第一階段的 Reranker 評測：" }));
        card.appendChild(
          h(
            "ul",
            {},
            r.children.map((c) => h("li", {}, h("a", { href: `/results/${c.id}`, text: c.label }), ` · Hit@5 ${fmt(c.metrics["hit@5"])}`))
          )
        );
      }
      root.appendChild(card);
    }
  }

  let redraw = null;
  let resizeTimer = null;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => redraw && redraw(), 150);
  });
  document.addEventListener("eb-theme", () => redraw && redraw());
  document.addEventListener("DOMContentLoaded", EB.guard(render));
})();
