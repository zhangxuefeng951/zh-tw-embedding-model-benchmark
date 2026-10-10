(function () {
  "use strict";
  const { h, $, $$, clear, api } = EB;

  const SAMPLE = {
    query: "臺灣最高的山是哪一座？",
    title: "玉山",
    text: "玉山是臺灣最高峰，主峰海拔 3,952 公尺，也是東北亞第一高峰，位於南投縣、嘉義縣與高雄市交界。",
    hasTitles: true,
    fromDataset: false,
  };

  const S = {
    providers: [],
    rerankers: [],
    presets: [],
    rPresets: [],
    datasets: [],
    // embedding
    preset: null,
    paramForm: null,
    pairs: [],
    samples: {},
    // reranker
    rPreset: null,
    rParamForm: null,
  };

  // ---------------------------------------------------------------- 分頁
  function showTab(name) {
    for (const b of $$(".tabs button")) b.classList.toggle("active", b.dataset.tab === name);
    for (const t of ["embedding", "rerank", "bm25"]) $("#tab-" + t).classList.toggle("hidden", t !== name);
    const p = EB.params();
    p.set("tab", name);
    history.replaceState(null, "", "?" + p.toString());
  }

  // ---------------------------------------------------------------- 共用
  function presetOptions(select, presets, selectedId) {
    clear(select);
    const user = presets.filter((p) => p.source === "user");
    const builtin = presets.filter((p) => p.source !== "user");
    if (user.length) select.appendChild(h("optgroup", { label: "自訂預設檔" }, user.map((p) => h("option", { value: p.id, text: p.name, selected: p.id === selectedId }))));
    select.appendChild(h("optgroup", { label: "內建預設檔" }, builtin.map((p) => h("option", { value: p.id, text: p.name, selected: p.id === selectedId }))));
  }

  function typeOptions(select, types, selected) {
    clear(select);
    const groups = { api: "API", llm: "LLM", local: "本地載入（需要 uv sync --extra local）" };
    for (const [g, label] of Object.entries(groups)) {
      const items = types.filter((t) => t.group === g);
      if (items.length) select.appendChild(h("optgroup", { label }, items.map((t) => h("option", { value: t.type, text: t.label, selected: t.type === selected }))));
    }
  }

  function presetInfo(container, preset) {
    clear(container);
    if (!preset) return;
    const notes = preset.notes || [];
    if (!preset.description && !notes.length) return;
    container.appendChild(
      h(
        "div",
        { class: "notice" + (notes.length ? " warn" : "") },
        preset.description ? h("div", { text: preset.description }) : null,
        notes.length ? h("ul", {}, notes.map((n) => h("li", { text: n }))) : null
      )
    );
  }

  function datasetChecklist(container, selected, name) {
    clear(container);
    const groups = {};
    for (const d of S.datasets) (groups[d.group] = groups[d.group] || []).push(d);
    for (const [g, items] of Object.entries(groups)) {
      const list = h("div", { class: "stack" });
      // C-MTEB 選項很多、成本也高，預設收合（有勾選時展開）
      if (g === "C-MTEB") {
        const open = items.some((d) => selected.includes(d.key));
        container.appendChild(
          h("details", { open, style: { margin: "14px 0 6px" } }, h("summary", { style: { cursor: "pointer", fontWeight: 650 }, text: `C-MTEB（${items.length} 個；文件庫約 10 萬筆，評測成本較高）` }), list)
        );
      } else {
        container.appendChild(h("h3", { text: g }));
        container.appendChild(list);
      }
      for (const d of items) {
        const status = d.prepared
          ? h("span", { class: "chip good", text: `已下載 · ${EB.fmtInt(d.num_queries)} 題 / ${EB.fmtInt(d.num_documents)} 文件${d.has_titles ? " · 有標題" : ""}` })
          : h("span", { class: "chip", text: `未下載（執行時自動下載，${d.size_hint || ""}）` });
        list.appendChild(
          h(
            "label",
            { class: "check", style: { display: "flex", alignItems: "flex-start" } },
            h("input", { type: "checkbox", name, value: d.key, checked: selected.includes(d.key) }),
            h("span", {}, h("strong", { text: d.name }), " ", status, h("span", { class: "field-help", style: { marginTop: 0 }, text: d.description }))
          )
        );
      }
    }
  }

  function checkedDatasets(name) {
    return $$(`input[name="${name}"]:checked`).map((i) => i.value);
  }

  function testResultBox(container, content) {
    clear(container);
    container.appendChild(h("div", { style: { marginTop: "14px" } }, content));
  }

  // ---------------------------------------------------------------- Embedding
  function providerSchema(type) {
    return S.providers.find((p) => p.type === type);
  }

  function renderParams(type, values) {
    const schema = providerSchema(type);
    $("#e-provider-desc").textContent = schema ? schema.description : "";
    S.paramForm = EB.paramForm($("#e-params"), schema, values);
    const modelInput = S.paramForm.inputs.model;
    if (modelInput) {
      const sync = () => ($("#e-label").placeholder = modelInput.value || "模型名稱");
      modelInput.addEventListener("input", sync);
      sync();
    }
    const dtype = S.paramForm.inputs.dtype;
    if (dtype) {
      const warn = () => {
        const isGemma = /embeddinggemma/i.test(S.paramForm.inputs.model.value || "");
        dtype.style.outline = isGemma && dtype.value === "float16" ? "2px solid var(--bad)" : "";
        if (isGemma && dtype.value === "float16") EB.toast("EmbeddingGemma 不要使用 float16（可能產生 NaN 或降低品質），請改用 bfloat16 或 float32", "error");
      };
      dtype.addEventListener("change", warn);
    }
  }

  function applyPreset(preset) {
    S.preset = preset;
    presetInfo($("#e-preset-info"), preset);
    $("#e-provider").value = preset.provider;
    renderParams(preset.provider, preset.params || {});
    S.pairs = (preset.prefix_pairs || []).map((p) => ({ label: p.label, query: p.query, document: p.document, checked: !!p.default }));
    renderPairs();
    const chips = clear($("#e-dim-chips"));
    const dims = preset.suggested_dims || [];
    if (dims.length) {
      chips.appendChild(document.createTextNode(" 建議："));
      for (const d of dims) {
        chips.appendChild(
          h("span", {
            class: "chip toggle",
            style: { marginLeft: "4px" },
            text: d,
            role: "button",
            onclick: () => {
              const set = new Set($("#e-dims").value.split(/[,，\s]+/).filter(Boolean).map(Number));
              set.has(d) ? set.delete(d) : set.add(d);
              $("#e-dims").value = [...set].sort((a, b) => b - a).join(",");
              updateSummary();
            },
          })
        );
      }
    }
    $("#e-dims").value = "";
    $("#e-label").value = "";
    updateSummary();
  }

  function onProviderChange() {
    const type = $("#e-provider").value;
    const current = S.paramForm ? S.paramForm.values() : {};
    let values;
    if (S.preset && type === S.preset.provider) values = { ...S.preset.params };
    else values = { model: current.model, ...((S.preset && S.preset.provider_params && S.preset.provider_params[type]) || {}) };
    renderParams(type, values);
  }

  function renderPairs() {
    const tbody = clear($("#e-pairs"));
    S.pairs.forEach((pair, i) => {
      const area = (field) => {
        const ta = h("textarea", { rows: 1, spellcheck: "false", placeholder: "（無前綴）", "aria-label": field === "query" ? "查詢前綴" : "文件前綴" });
        ta.value = pair[field];
        ta.addEventListener("input", () => {
          pair[field] = ta.value;
          renderPreview();
        });
        return ta;
      };
      tbody.appendChild(
        h(
          "tr",
          {},
          h(
            "td",
            {},
            h("input", {
              type: "checkbox",
              checked: pair.checked,
              "aria-label": "使用這組前綴",
              onchange: (e) => {
                pair.checked = e.target.checked;
                renderPreview();
                updateSummary();
              },
            })
          ),
          h(
            "td",
            {},
            h("input", {
              type: "text",
              value: pair.label,
              placeholder: "組合名稱",
              oninput: (e) => {
                pair.label = e.target.value;
                renderPreview();
              },
            })
          ),
          h("td", {}, area("query")),
          h("td", {}, area("document")),
          h(
            "td",
            {},
            h("button", {
              type: "button",
              class: "small",
              title: "刪除",
              text: "✕",
              onclick: () => {
                S.pairs.splice(i, 1);
                renderPairs();
              },
            })
          )
        )
      );
    });
    renderPreview();
    updateSummary();
  }

  async function sampleFor(key) {
    if (!key) return SAMPLE;
    const d = S.datasets.find((x) => x.key === key);
    if (!d || !d.prepared) return SAMPLE;
    if (!S.samples[key]) {
      try {
        const s = await api(`/api/datasets/${encodeURIComponent(key)}/sample?n=1`);
        const item = s.items[0];
        S.samples[key] = { query: item.query, title: item.gold[0].title, text: item.gold[0].text, hasTitles: s.has_titles, fromDataset: true, name: d.name };
      } catch {
        return SAMPLE;
      }
    }
    return S.samples[key];
  }

  function previewLine(template, text, title) {
    const r = EB.renderPrefix(template, text, title);
    const box = h("div", { class: "preview" });
    if (r.prefix === null) box.appendChild(document.createTextNode(r.rendered));
    else {
      if (r.prefix) box.appendChild(EB.prefixNode(r.prefix));
      const shown = text.length > 90 ? text.slice(0, 90) + "…" : text;
      box.appendChild(document.createTextNode(shown));
    }
    return box;
  }

  let previewToken = 0;
  async function renderPreview() {
    const token = ++previewToken;
    const key = checkedDatasets("e-ds")[0];
    const sample = await sampleFor(key);
    if (token !== previewToken) return;
    const box = clear($("#e-preview"));
    const checked = S.pairs.filter((p) => p.checked);
    if (!checked.length) {
      box.appendChild(h("div", { class: "notice warn", text: "請至少勾選一組前綴組合（不加前綴也請保留一組空白的組合）。" }));
      return;
    }
    box.appendChild(
      h("p", { class: "small muted", text: sample.fromDataset ? `範例取自「${sample.name}」的第一題與其正解文件。` : "範例文字（選擇已下載的資料集後會改用該資料集的第一題）。" })
    );
    const seen = new Map();
    const warnings = [];
    for (const p of checked) {
      const k = JSON.stringify([p.query, p.document]);
      if (seen.has(k)) warnings.push(`「${p.label || "未命名"}」與「${seen.get(k)}」的前綴完全相同，只會執行一次。`);
      else seen.set(k, p.label || "未命名");
      if ((p.document || "").includes("{title}") && !sample.hasTitles) warnings.push(`「${p.label}」使用 {title}，但目前的資料集沒有標題，會代入 none。`);
      if (/\{(?!title\}|text\})[^}]*\}/.test(p.query + p.document)) warnings.push(`「${p.label}」含有不認得的 {…} 標記，會被當成一般文字送出。`);
      box.appendChild(
        h(
          "div",
          { style: { marginBottom: "12px" } },
          h("strong", { text: p.label || "未命名組合" }),
          h("div", { class: "preview-label", text: "查詢" }),
          previewLine(p.query, sample.query, null),
          h("div", { class: "preview-label", text: "文件" }),
          previewLine(p.document, sample.text, sample.title)
        )
      );
    }
    if (warnings.length) box.insertBefore(h("div", { class: "notice warn" }, h("ul", {}, warnings.map((w) => h("li", { text: w })))), box.firstChild);
  }

  function updateSummary() {
    const pairs = S.pairs.filter((p) => p.checked).length;
    const ds = checkedDatasets("e-ds").length;
    const dims = $("#e-dims").value.split(/[,，\s]+/).filter(Boolean).length || 1;
    $("#e-summary").textContent = `${ds} 個資料集 × ${pairs} 組前綴 × ${dims} 種維度 = ${ds * pairs * dims} 個結果`;
  }

  function embeddingConfig() {
    const params = S.paramForm.values();
    const pairs = S.pairs.filter((p) => p.checked).map((p) => ({ label: p.label, query: p.query, document: p.document }));
    const dims = $("#e-dims")
      .value.split(/[,，\s]+/)
      .filter(Boolean)
      .map((x) => {
        const n = parseInt(x, 10);
        if (!(n > 0)) throw new Error(`評測維度格式錯誤：${x}`);
        return n;
      });
    const limit = $("#e-limit").value.trim();
    return {
      label: $("#e-label").value.trim(),
      preset_id: S.preset ? S.preset.id : null,
      provider: $("#e-provider").value,
      params,
      prefix_pairs: pairs,
      datasets: checkedDatasets("e-ds"),
      eval_dims: dims,
      query_limit: limit ? parseInt(limit, 10) : null,
    };
  }

  function longestText(longest) {
    if (!longest) return "";
    if (longest.error) return `資料集最長的文件（${longest.chars} 字）超過模型長度上限。`;
    if (longest.truncated) {
      const how = (longest.notices || []).find((n) => n.startsWith("使用 tokenizer")) ? "（以 tokenizer 精確截斷）" : "";
      return `資料集最長的文件（${longest.chars} 字）超過長度上限，會自動截短為前 ${longest.kept_chars} 字${how}。`;
    }
    return `資料集最長的文件（${longest.chars} 字）也能正常送出。`;
  }

  async function testEmbedding() {
    const cfg = embeddingConfig();
    const box = $("#e-test-result");
    testResultBox(box, h("div", { class: "notice" }, h("span", { class: "spinner" }), " 測試中…（本地模型第一次載入可能需要一些時間）"));
    const out = await api("/api/test/embedding", {
      method: "POST",
      body: { provider: cfg.provider, params: cfg.params, prefix_pairs: cfg.prefix_pairs.length ? cfg.prefix_pairs : [{ label: "無前綴", query: "", document: "" }], dataset: cfg.datasets[0] },
    });
    if (!out.ok) {
      testResultBox(box, h("div", { class: "notice bad" }, h("strong", { text: "連線失敗：" }), h("pre", { text: out.error })));
      return;
    }
    const rows = out.items.map((it) =>
      h(
        "tr",
        {},
        h("td", { text: it.label }),
        h("td", { class: "num", text: it.dims }),
        h("td", { class: "num", text: EB.fmt(it.cosine) }),
        h("td", { class: "num", text: `${EB.fmt(it.query_norm, 3)} / ${EB.fmt(it.document_norm, 3)}` }),
        h("td", {}, it.nan ? h("span", { class: "chip bad", text: "有 NaN" }) : h("span", { class: "chip good", text: "正常" })),
        h("td", { class: "num", text: it.elapsed_ms + " ms" })
      )
    );
    const first = out.items[0];
    testResultBox(
      box,
      h(
        "div",
        {},
        h("div", { class: "notice good", text: `連線成功（建立連線 / 載入模型 ${out.load_seconds} 秒）。` + longestText(out.longest) }),
        out.warnings.length ? h("div", { class: "notice warn" }, h("ul", {}, out.warnings.map((w) => h("li", { text: w })))) : null,
        h(
          "div",
          { class: "table-wrap" },
          h(
            "table",
            {},
            h("thead", {}, h("tr", {}, ["前綴組合", "維度", "查詢↔正解 cosine", "向量長度（查詢 / 文件）", "數值", "耗時"].map((t) => h("th", { text: t })))),
            h("tbody", {}, rows)
          )
        ),
        first ? h("div", { class: "preview-label", text: `「${first.label}」實際送出的查詢` }) : null,
        first ? h("pre", { class: "preview", text: first.query_text }) : null,
        first ? h("div", { class: "preview-label", text: "實際送出的文件" }) : null,
        first ? h("pre", { class: "preview", text: first.document_text }) : null
      )
    );
  }

  async function saveEmbeddingPreset() {
    const cfg = embeddingConfig();
    const base = S.preset;
    const overwrite = base && base.source === "user" && confirm(`要覆寫自訂預設檔「${base.name}」嗎？\n按「取消」則另存成新的預設檔。`);
    const name = overwrite ? base.name : await EB.promptDialog("另存為自訂預設檔", "名稱", base ? base.name + "（自訂）" : cfg.params.model);
    if (!name) return;
    const preset = {
      name,
      description: base ? base.description : "",
      provider: cfg.provider,
      params: cfg.params,
      provider_params: base ? base.provider_params : {},
      prefix_pairs: S.pairs.map((p) => ({ label: p.label, query: p.query, document: p.document, default: p.checked })),
      suggested_dims: cfg.eval_dims.length ? cfg.eval_dims : (base && base.suggested_dims) || [],
      notes: base ? base.notes : [],
    };
    const saved = await api("/api/presets", { method: "POST", body: { kind: "embedding", preset, id: overwrite ? base.id : null } });
    S.presets = await api("/api/presets?kind=embedding");
    presetOptions($("#e-preset"), S.presets, saved.id);
    S.preset = S.presets.find((p) => p.id === saved.id);
    EB.toast(`已儲存自訂預設檔「${saved.name}」`);
  }

  async function submitEmbedding() {
    const cfg = embeddingConfig();
    if (!cfg.prefix_pairs.length) throw new Error("請至少勾選一組前綴組合");
    if (!cfg.datasets.length) throw new Error("請至少選擇一個資料集");
    const job = await api("/api/jobs", { method: "POST", body: { kind: "embedding", config: cfg } });
    EB.toast("已加入任務佇列");
    location.href = "/jobs?highlight=" + job.id;
  }

  // ---------------------------------------------------------------- Reranker
  function rerankerSchema(type) {
    return S.rerankers.find((r) => r.type === type);
  }

  function renderRParams(type, values) {
    const schema = rerankerSchema(type);
    $("#r-type-desc").textContent = schema ? schema.description : "";
    S.rParamForm = EB.paramForm($("#r-params"), schema, values);
  }

  function applyRPreset(preset) {
    S.rPreset = preset;
    presetInfo($("#r-preset-info"), preset);
    $("#r-type").value = preset.reranker;
    renderRParams(preset.reranker, preset.params || {});
    $("#r-candidates").value = preset.candidates || 50;
  }

  function onRTypeChange() {
    const type = $("#r-type").value;
    const current = S.rParamForm ? S.rParamForm.values() : {};
    let values;
    if (S.rPreset && type === S.rPreset.reranker) values = { ...S.rPreset.params };
    else values = { model: current.model, ...((S.rPreset && S.rPreset.provider_params && S.rPreset.provider_params[type]) || {}) };
    renderRParams(type, values);
  }

  async function loadSources(datasetKey, selectedId) {
    const sel = clear($("#r-source"));
    if (!datasetKey) {
      sel.appendChild(h("option", { value: "", text: "（還沒有任何評測結果，請先跑 embedding 或 BM25）" }));
      return;
    }
    const rows = await api(`/api/results?dataset=${encodeURIComponent(datasetKey)}&ks=5,10,${$("#r-candidates").value || 50}`);
    rows.sort((a, b) => (b.metrics["hit@5"] ?? 0) - (a.metrics["hit@5"] ?? 0));
    for (const r of rows) {
      const n = parseInt($("#r-candidates").value, 10) || 50;
      const recall = r.metrics[`hit@${n}`];
      const parts = [r.label, r.pair_label, r.dims ? `${r.dims} 維` : null, `Hit@5 ${EB.fmt(r.metrics["hit@5"])}`, recall !== undefined ? `Hit@${n} ${EB.fmt(recall)}` : null, r.query_limit ? `試跑 ${r.num_queries} 題` : null];
      sel.appendChild(h("option", { value: r.id, text: parts.filter(Boolean).join(" · "), selected: r.id === selectedId }));
    }
    if (!rows.length) sel.appendChild(h("option", { value: "", text: "（這個資料集沒有結果）" }));
  }

  async function testReranker() {
    const box = $("#r-test-result");
    testResultBox(box, h("div", { class: "notice" }, h("span", { class: "spinner" }), " 測試中…"));
    const out = await api("/api/test/reranker", {
      method: "POST",
      body: { reranker: $("#r-type").value, params: S.rParamForm.values(), dataset: $("#r-dataset").value },
    });
    if (!out.ok) {
      testResultBox(box, h("div", { class: "notice bad" }, h("strong", { text: "連線失敗：" }), h("pre", { text: out.error })));
      return;
    }
    testResultBox(
      box,
      h(
        "div",
        {},
        h("div", { class: "notice good", text: `連線成功（${out.elapsed_ms} ms）` }),
        out.warnings.length ? h("div", { class: "notice warn" }, h("ul", {}, out.warnings.map((w) => h("li", { text: w })))) : null,
        h("div", { class: "preview-label", text: "查詢" }),
        h("pre", { class: "preview", text: out.query }),
        h(
          "table",
          { style: { marginTop: "8px" } },
          h("thead", {}, h("tr", {}, h("th", { text: "文件" }), h("th", { class: "num", text: "分數" }))),
          h(
            "tbody",
            {},
            out.documents.map((d, i) => h("tr", {}, h("td", { text: (i === 0 ? "【正解】" : "【無關】") + d.slice(0, 80) }), h("td", { class: "num", text: EB.fmt(out.scores[i]) })))
          )
        )
      )
    );
  }

  async function saveRerankerPreset() {
    const base = S.rPreset;
    const overwrite = base && base.source === "user" && confirm(`要覆寫自訂預設檔「${base.name}」嗎？\n按「取消」則另存成新的預設檔。`);
    const params = S.rParamForm.values();
    const name = overwrite ? base.name : await EB.promptDialog("另存為自訂預設檔", "名稱", base ? base.name + "（自訂）" : params.model);
    if (!name) return;
    const preset = {
      name,
      description: base ? base.description : "",
      reranker: $("#r-type").value,
      params,
      provider_params: base ? base.provider_params : {},
      candidates: parseInt($("#r-candidates").value, 10) || 50,
      notes: base ? base.notes : [],
    };
    const saved = await api("/api/presets", { method: "POST", body: { kind: "reranker", preset, id: overwrite ? base.id : null } });
    S.rPresets = await api("/api/presets?kind=reranker");
    presetOptions($("#r-preset"), S.rPresets, saved.id);
    S.rPreset = S.rPresets.find((p) => p.id === saved.id);
    EB.toast(`已儲存自訂預設檔「${saved.name}」`);
  }

  async function submitReranker() {
    const source = $("#r-source").value;
    if (!source) throw new Error("請選擇第一階段結果");
    const cfg = {
      source_result_id: source,
      reranker: $("#r-type").value,
      params: S.rParamForm.values(),
      candidates: parseInt($("#r-candidates").value, 10) || 50,
      include_title: $("#r-include-title").checked,
      label: $("#r-label").value.trim(),
      preset_id: S.rPreset ? S.rPreset.id : null,
    };
    const job = await api("/api/jobs", { method: "POST", body: { kind: "rerank", config: cfg } });
    EB.toast("已加入任務佇列");
    location.href = "/jobs?highlight=" + job.id;
  }

  // ---------------------------------------------------------------- BM25
  async function submitBm25() {
    const datasets = checkedDatasets("b-ds");
    if (!datasets.length) throw new Error("請至少選擇一個資料集");
    const limit = $("#b-limit").value.trim();
    const cfg = { datasets, k1: parseFloat($("#b-k1").value), b: parseFloat($("#b-b").value), query_limit: limit ? parseInt(limit, 10) : null };
    const job = await api("/api/jobs", { method: "POST", body: { kind: "bm25", config: cfg } });
    EB.toast("已加入任務佇列");
    location.href = "/jobs?highlight=" + job.id;
  }

  // ---------------------------------------------------------------- 初始化
  async function init() {
    const params = EB.params();
    [S.providers, S.rerankers, S.presets, S.rPresets, S.datasets] = await Promise.all([
      api("/api/providers?kind=embedding"),
      api("/api/providers?kind=reranker"),
      api("/api/presets?kind=embedding"),
      api("/api/presets?kind=reranker"),
      api("/api/datasets"),
    ]);

    for (const b of $$(".tabs button")) b.addEventListener("click", () => showTab(b.dataset.tab));

    // Embedding
    const wanted = params.get("preset");
    const first = S.presets.find((p) => p.id === wanted) || S.presets.find((p) => p.id === "embeddinggemma-2") || S.presets[0];
    presetOptions($("#e-preset"), S.presets, first.id);
    typeOptions($("#e-provider"), S.providers, first.provider);
    $("#e-preset").addEventListener("change", () => applyPreset(S.presets.find((p) => p.id === $("#e-preset").value)));
    $("#e-provider").addEventListener("change", EB.guard(onProviderChange));
    datasetChecklist($("#e-datasets"), ["drcd-test"], "e-ds");
    $("#e-datasets").addEventListener("change", () => {
      renderPreview();
      updateSummary();
    });
    $("#e-dims").addEventListener("input", updateSummary);
    $("#e-add-pair").addEventListener("click", () => {
      S.pairs.push({ label: `自訂組合 ${S.pairs.length + 1}`, query: "", document: "", checked: true });
      renderPairs();
    });
    $("#e-test").addEventListener("click", (e) => EB.busy(e.target, testEmbedding));
    $("#e-save").addEventListener("click", (e) => EB.busy(e.target, saveEmbeddingPreset));
    $("#e-submit").addEventListener("click", (e) => EB.busy(e.target, submitEmbedding));
    applyPreset(first);

    // Reranker
    const rFirst = S.rPresets.find((p) => p.id === params.get("rpreset")) || S.rPresets[0];
    presetOptions($("#r-preset"), S.rPresets, rFirst.id);
    typeOptions($("#r-type"), S.rerankers, rFirst.reranker);
    $("#r-preset").addEventListener("change", () => applyRPreset(S.rPresets.find((p) => p.id === $("#r-preset").value)));
    $("#r-type").addEventListener("change", EB.guard(onRTypeChange));
    applyRPreset(rFirst);
    const resultSets = await api("/api/results/datasets");
    const rds = $("#r-dataset");
    let sourceId = params.get("source");
    let sourceDataset = null;
    if (sourceId) {
      try {
        sourceDataset = (await api(`/api/results/${sourceId}`)).dataset_key;
      } catch {
        sourceId = null;
      }
    }
    for (const d of resultSets) rds.appendChild(h("option", { value: d.dataset_key, text: `${d.name}（${d.n}）`, selected: d.dataset_key === sourceDataset }));
    rds.addEventListener("change", EB.guard(() => loadSources(rds.value)));
    $("#r-candidates").addEventListener("change", EB.guard(() => loadSources(rds.value, $("#r-source").value)));
    await loadSources(rds.value, sourceId);
    $("#r-test").addEventListener("click", (e) => EB.busy(e.target, testReranker));
    $("#r-save").addEventListener("click", (e) => EB.busy(e.target, saveRerankerPreset));
    $("#r-submit").addEventListener("click", (e) => EB.busy(e.target, submitReranker));

    // BM25
    datasetChecklist($("#b-datasets"), ["drcd-test"], "b-ds");
    $("#b-submit").addEventListener("click", (e) => EB.busy(e.target, submitBm25));

    showTab(params.get("tab") || (sourceId ? "rerank" : "embedding"));
  }

  document.addEventListener("DOMContentLoaded", EB.guard(init));
})();
