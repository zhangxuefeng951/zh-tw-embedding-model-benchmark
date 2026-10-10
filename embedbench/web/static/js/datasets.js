(function () {
  "use strict";
  const { h, $, clear, api } = EB;

  async function preview(d) {
    const data = await api(`/api/datasets/${encodeURIComponent(d.key)}/sample?n=5`);
    $("#preview-card").classList.remove("hidden");
    $("#preview-title").textContent = `預覽：${d.name}`;
    const box = clear($("#preview"));
    box.appendChild(h("p", { class: "small secondary", text: `${EB.fmtInt(data.num_queries)} 題、${EB.fmtInt(data.num_documents)} 篇文件${data.has_titles ? "，有文件標題" : "，沒有文件標題"}。以下是前 5 題與其正解文件：` }));
    for (const item of data.items) {
      box.appendChild(
        h(
          "div",
          { class: "qitem" },
          h("div", { class: "qtext", text: item.query }),
          h(
            "ul",
            { class: "doclist" },
            item.gold.map((g) => h("li", { class: "relevant" }, h("span", { class: "rank", text: "正解" }), h("span", {}, g.title ? h("span", { class: "doc-title", text: g.title }) : null, h("span", { class: "snippet", text: g.text })), h("span")))
          )
        )
      );
    }
    $("#preview-card").scrollIntoView({ behavior: "smooth" });
  }

  async function render() {
    const list = await api("/api/datasets");
    const table = clear($("#table"));
    table.appendChild(h("thead", {}, h("tr", {}, ["資料集", "狀態", "題數", "文件數", "來源 / 授權", ""].map((t) => h("th", { text: t })))));
    const tbody = h("tbody");
    let group = null;
    for (const d of list) {
      if (d.group !== group) {
        group = d.group;
        tbody.appendChild(h("tr", {}, h("td", { colspan: 6, style: { background: "var(--surface-2)", fontWeight: 600 }, text: group })));
      }
      const actions = [];
      if (d.prepared) actions.push(h("button", { type: "button", class: "small", text: "預覽", onclick: (e) => EB.busy(e.target, () => preview(d)) }));
      if (d.builtin)
        actions.push(
          h("button", {
            type: "button",
            class: "small",
            text: d.prepared ? "重新下載" : "下載",
            onclick: (e) =>
              EB.busy(e.target, async () => {
                const job = await api(`/api/datasets/${encodeURIComponent(d.key)}/prepare`, { method: "POST", body: { force: d.prepared } });
                EB.toast("已加入任務佇列，可在「任務」頁查看進度");
                setTimeout(render, 1500);
                return job;
              }),
          })
        );
      if (d.prepared)
        actions.push(
          h("button", {
            type: "button",
            class: "small danger",
            text: d.builtin ? "清除" : "刪除",
            onclick: (e) =>
              EB.busy(e.target, async () => {
                const msg = d.builtin ? `清除「${d.name}」的下載檔案？（下次使用時會重新下載）` : `刪除自訂資料集「${d.name}」？已完成的評測結果會保留，但無法再查看逐題分析。`;
                if (!confirm(msg)) return;
                await api(`/api/datasets/${encodeURIComponent(d.key)}`, { method: "DELETE" });
                render();
              }),
          })
        );
      tbody.appendChild(
        h(
          "tr",
          {},
          h("td", { style: { maxWidth: "460px" } }, h("strong", { text: d.name }), h("span", { class: "sub mono", text: d.key }), h("span", { class: "sub", text: d.description })),
          h("td", {}, d.prepared ? h("span", { class: "chip good", text: "已下載" }) : h("span", { class: "chip", text: "未下載" }), d.has_titles ? h("span", { class: "sub", text: "有標題" }) : null),
          h("td", { class: "num", text: d.prepared ? EB.fmtInt(d.num_queries) : "—" }),
          h("td", { class: "num", text: d.prepared ? EB.fmtInt(d.num_documents) : d.size_hint || "—" }),
          h("td", { class: "small" }, d.source && d.source.startsWith("http") ? h("a", { href: d.source, target: "_blank", rel: "noopener", text: "來源" }) : d.source || "", d.license ? h("span", { class: "sub", text: d.license }) : null),
          h("td", {}, h("div", { class: "row", style: { gap: "6px", justifyContent: "flex-end" } }, actions))
        )
      );
    }
    table.appendChild(tbody);
  }

  function bindUpload(formId) {
    const form = $(formId);
    form.addEventListener("submit", (e) => {
      e.preventDefault();
      const btn = form.querySelector("button[type=submit]");
      EB.busy(btn, async () => {
        const ds = await api("/api/datasets/upload", { method: "POST", form: new FormData(form) });
        EB.toast(`已匯入「${ds.name}」：${ds.num_queries} 題、${ds.num_documents} 篇文件`);
        form.reset();
        await render();
      });
    });
  }

  document.addEventListener(
    "DOMContentLoaded",
    EB.guard(async () => {
      $("#preview-close").addEventListener("click", () => $("#preview-card").classList.add("hidden"));
      bindUpload("#simple-form");
      bindUpload("#beir-form");
      await render();
    })
  );
})();
