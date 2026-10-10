(function () {
  "use strict";
  const { h, $, $$, clear, api } = EB;
  let kind = "embedding";
  let presets = [];

  function card(p) {
    const type = p.provider || p.reranker;
    const pairs = p.prefix_pairs || [];
    const actions = [
      h("a", { class: "btn small primary", href: kind === "embedding" ? `/new?preset=${encodeURIComponent(p.id)}` : `/new?tab=rerank&rpreset=${encodeURIComponent(p.id)}`, text: "用這個預設評測" }),
      h("button", {
        type: "button",
        class: "small",
        text: "匯出 YAML",
        onclick: EB.guard(async () => EB.textDialog(`${p.name}（YAML）`, await api(`/api/presets/${encodeURIComponent(p.id)}/yaml?kind=${kind}`))),
      }),
    ];
    if (p.source === "user")
      actions.push(
        h("button", {
          type: "button",
          class: "small danger",
          text: "刪除",
          onclick: EB.guard(async () => {
            if (!confirm(`刪除自訂預設檔「${p.name}」？`)) return;
            await api(`/api/presets/${encodeURIComponent(p.id)}`, { method: "DELETE" });
            await load();
          }),
        })
      );
    return h(
      "div",
      { class: "qitem" },
      h(
        "div",
        { class: "row" },
        h("strong", { text: p.name }),
        h("span", { class: "chip " + (p.source === "user" ? "accent" : ""), text: p.source === "user" ? "自訂" : "內建" }),
        h("span", { class: "chip", text: type }),
        p.params && p.params.model ? h("span", { class: "chip mono", text: p.params.model }) : null,
        h("span", { class: "spacer" }),
        actions
      ),
      p.description ? h("p", { class: "small secondary", style: { margin: "6px 0 0" }, text: p.description }) : null,
      pairs.length
        ? h(
            "table",
            { style: { marginTop: "8px" } },
            h("thead", {}, h("tr", {}, h("th", { text: "前綴組合" }), h("th", { text: "查詢前綴" }), h("th", { text: "文件前綴" }))),
            h(
              "tbody",
              {},
              pairs.map((pair) =>
                h(
                  "tr",
                  {},
                  h("td", {}, pair.label, pair.default ? h("span", { class: "chip good", style: { marginLeft: "6px" }, text: "預設" }) : null),
                  h("td", { class: "mono" }, EB.prefixNode(pair.query)),
                  h("td", { class: "mono" }, EB.prefixNode(pair.document))
                )
              )
            )
          )
        : null,
      p.notes && p.notes.length ? h("ul", { class: "small muted", style: { margin: "8px 0 0" } }, p.notes.map((n) => h("li", { text: n }))) : null
    );
  }

  function render() {
    const q = $("#search").value.trim().toLowerCase();
    const items = presets.filter((p) => !q || JSON.stringify(p).toLowerCase().includes(q));
    const list = clear($("#list"));
    if (!items.length) list.appendChild(h("div", { class: "empty", text: "沒有符合的預設檔" }));
    items.forEach((p) => list.appendChild(card(p)));
  }

  async function load() {
    presets = await api(`/api/presets?kind=${kind}`);
    render();
  }

  document.addEventListener(
    "DOMContentLoaded",
    EB.guard(async () => {
      for (const b of $$(".tabs button"))
        b.addEventListener(
          "click",
          EB.guard(async () => {
            kind = b.dataset.kind;
            for (const x of $$(".tabs button")) x.classList.toggle("active", x === b);
            await load();
          })
        );
      $("#search").addEventListener("input", render);
      await load();
    })
  );
})();
