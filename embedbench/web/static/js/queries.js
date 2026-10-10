(function () {
  "use strict";
  const { h, $, clear, api, fmt } = EB;
  const id = window.RESULT_ID;
  const state = { k: 5, status: "miss", sort: "index", q: "", page: 1, perPage: 20 };

  function snippet(text, title) {
    const el = h("span", { class: "snippet expandable clamped", title: "點擊展開 / 收合", text });
    el.addEventListener("click", () => el.classList.toggle("clamped"));
    return h("span", {}, title ? h("span", { class: "doc-title", text: title }) : null, el);
  }

  function rankChip(rank, k) {
    if (rank > 1e9) return h("span", { class: "chip bad", text: "沒有正解" });
    return h("span", { class: "chip " + (rank <= k ? "good" : "bad"), text: rank <= k ? `命中：第 ${rank} 名` : `未命中：正解第 ${rank} 名` });
  }

  function renderItem(item) {
    const goldIds = new Set(item.gold.map((g) => g.index));
    return h(
      "div",
      { class: "qitem" },
      h("div", { class: "qhead" }, h("span", { class: "muted small", text: `#${item.position + 1}` }), h("span", { class: "qtext", text: item.query }), rankChip(item.best_rank, state.k)),
      h(
        "div",
        { class: "gold" },
        h("span", { class: "field-label", text: "正解文件" }),
        h(
          "ul",
          { class: "doclist" },
          item.gold.map((g) => h("li", { class: "relevant" }, h("span", { class: "rank", text: `#${g.rank}` }), snippet(g.text, g.title), h("span", { class: "score", text: g.gain !== 1 ? `相關度 ${g.gain}` : "" })))
        )
      ),
      h(
        "div",
        { class: "gold" },
        h("span", { class: "field-label", text: `模型排的前 ${item.top.length} 名` }),
        h(
          "ul",
          { class: "doclist" },
          item.top.map((t) =>
            h("li", { class: goldIds.has(t.index) ? "relevant" : "" }, h("span", { class: "rank", text: `${t.rank}.` }), snippet(t.text, t.title), h("span", { class: "score", text: t.score === null ? "" : fmt(t.score) }))
          )
        )
      )
    );
  }

  function pager(total) {
    const box = clear($("#pager"));
    const pages = Math.max(1, Math.ceil(total / state.perPage));
    if (pages <= 1) return;
    const go = (p) => () => {
      state.page = p;
      load();
      window.scrollTo({ top: 0 });
    };
    box.appendChild(h("button", { type: "button", class: "small", text: "上一頁", disabled: state.page <= 1, onclick: go(state.page - 1) }));
    box.appendChild(h("span", { class: "small secondary", text: `第 ${state.page} / ${pages} 頁` }));
    box.appendChild(h("button", { type: "button", class: "small", text: "下一頁", disabled: state.page >= pages, onclick: go(state.page + 1) }));
  }

  let token = 0;
  async function load() {
    const t = ++token;
    const qs = new URLSearchParams({ k: state.k, status: state.status, sort: state.sort, q: state.q, page: state.page, per_page: state.perPage });
    const data = await api(`/api/results/${id}/queries?${qs}`);
    if (t !== token) return;
    const pct = data.num_queries ? ((data.misses / data.num_queries) * 100).toFixed(1) : "0";
    $("#summary").textContent = `共 ${EB.fmtInt(data.num_queries)} 題，K=${state.k} 時未命中 ${EB.fmtInt(data.misses)} 題（${pct}%）。目前條件符合 ${EB.fmtInt(data.total)} 題。綠底為正解文件，點擊文字可以展開全文。`;
    const box = clear($("#items"));
    if (!data.items.length) box.appendChild(h("div", { class: "empty", text: "沒有符合條件的題目。" }));
    data.items.forEach((item) => box.appendChild(renderItem(item)));
    pager(data.total);
  }

  async function init() {
    const r = await api(`/api/results/${id}?ks=5`);
    $("#subtitle").textContent = `${r.label} · ${r.pair_label || ""}${r.dims ? ` · ${r.dims} 維` : ""}`;
    const ksel = $("#k");
    for (const k of [1, 3, 5, 10, 20, 50, 100]) ksel.appendChild(h("option", { value: k, text: k, selected: k === state.k }));
    const bind = (sel, key, parse) =>
      $(sel).addEventListener(
        "change",
        EB.guard(async () => {
          state[key] = parse ? parse($(sel).value) : $(sel).value;
          state.page = 1;
          await load();
        })
      );
    bind("#k", "k", Number);
    bind("#status", "status");
    bind("#sort", "sort");
    let timer = null;
    $("#search").addEventListener("input", () => {
      clearTimeout(timer);
      timer = setTimeout(
        EB.guard(async () => {
          state.q = $("#search").value;
          state.page = 1;
          await load();
        }),
        300
      );
    });
    await load();
  }

  document.addEventListener("DOMContentLoaded", EB.guard(init));
})();
