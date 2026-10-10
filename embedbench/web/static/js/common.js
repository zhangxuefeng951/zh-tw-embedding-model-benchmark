/* 共用工具：API、DOM 建構（一律用 textContent，不用 innerHTML 插入資料）、通知、主題、參數表單。 */
(function () {
  "use strict";

  const EB = (window.EB = {});

  // ---- API -----------------------------------------------------------------
  EB.api = async function (path, opts = {}) {
    const init = { method: opts.method || "GET", headers: {} };
    if (opts.form) {
      init.body = opts.form;
    } else if (opts.body !== undefined) {
      init.headers["Content-Type"] = "application/json";
      init.body = JSON.stringify(opts.body);
    }
    const resp = await fetch(path, init);
    const type = resp.headers.get("Content-Type") || "";
    const data = type.includes("application/json") ? await resp.json() : await resp.text();
    if (!resp.ok) {
      const msg = (data && data.error) || (typeof data === "string" ? data : resp.statusText);
      throw new Error(msg);
    }
    return data;
  };

  // ---- DOM -----------------------------------------------------------------
  EB.h = function (tag, attrs, ...children) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v === null || v === undefined || v === false) continue;
      if (k === "class") el.className = v;
      else if (k === "text") el.textContent = v;
      else if (k === "style" && typeof v === "object") Object.assign(el.style, v);
      else if (k === "dataset") Object.assign(el.dataset, v);
      else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
      else if (k === "value") el.value = v;
      else if (k === "checked" || k === "disabled" || k === "selected" || k === "open") el[k] = !!v;
      else el.setAttribute(k, v === true ? "" : v);
    }
    EB.append(el, children);
    return el;
  };

  EB.append = function (el, children) {
    for (const c of children.flat(Infinity)) {
      if (c === null || c === undefined || c === false) continue;
      el.appendChild(c instanceof Node ? c : document.createTextNode(String(c)));
    }
    return el;
  };

  EB.clear = function (el) {
    while (el.firstChild) el.removeChild(el.firstChild);
    return el;
  };

  EB.$ = (sel, root) => (root || document).querySelector(sel);
  EB.$$ = (sel, root) => Array.from((root || document).querySelectorAll(sel));

  // ---- 格式化 --------------------------------------------------------------
  EB.fmt = function (v, digits = 4) {
    if (v === null || v === undefined || Number.isNaN(v)) return "—";
    return Number(v).toFixed(digits);
  };
  EB.fmtInt = (v) => (v === null || v === undefined ? "—" : Number(v).toLocaleString("zh-TW"));
  EB.fmtTime = function (iso) {
    if (!iso) return "—";
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return iso;
    return d.toLocaleString("zh-TW", { hour12: false });
  };
  EB.duration = function (start, end) {
    if (!start) return "";
    const s = Math.max(0, ((end ? new Date(end) : new Date()) - new Date(start)) / 1000);
    if (s < 60) return `${Math.round(s)} 秒`;
    if (s < 3600) return `${Math.floor(s / 60)} 分 ${Math.round(s % 60)} 秒`;
    return `${Math.floor(s / 3600)} 小時 ${Math.round((s % 3600) / 60)} 分`;
  };
  EB.metricLabel = function (key) {
    const [m, k] = key.split("@");
    return ({ hit: "Hit", mrr: "MRR", ndcg: "NDCG", recall: "Recall" }[m] || m) + "@" + k;
  };
  EB.parseKs = function (text) {
    return Array.from(
      new Set(
        String(text || "")
          .split(/[,，\s]+/)
          .map((x) => parseInt(x, 10))
          .filter((x) => x > 0 && x <= 100)
      )
    ).sort((a, b) => a - b);
  };
  EB.params = () => new URLSearchParams(location.search);

  /* 顯示前綴：換行顯示為 ↵、尾端空白顯示為 ␣，讓「看不見的字元」看得見。 */
  EB.prefixNode = function (template, opts = {}) {
    const span = EB.h("span", { class: opts.mark === false ? "" : "prefix-mark" });
    if (!template) {
      span.className = "muted";
      span.textContent = opts.emptyText || "（無前綴）";
      return span;
    }
    const parts = template.split("\n");
    parts.forEach((part, i) => {
      let body = part;
      let trailing = "";
      if (i === parts.length - 1) {
        const m = part.match(/^(.*?)( +)$/s);
        if (m) {
          body = m[1];
          trailing = m[2];
        }
      }
      span.appendChild(document.createTextNode(body));
      if (trailing) span.appendChild(EB.h("span", { class: "ws", text: "␣".repeat(trailing.length) }));
      if (i < parts.length - 1) {
        span.appendChild(EB.h("span", { class: "ws", text: "↵" }));
        span.appendChild(document.createTextNode("\n"));
      }
    });
    return span;
  };

  /* 和後端 prefixes.render 相同的規則：{title} 沒有標題時代入 none；沒有 {text} 時接在後面。 */
  EB.renderPrefix = function (template, text, title) {
    template = template || "";
    const t = title && title.trim() ? title.trim() : "none";
    const hasText = template.includes("{text}");
    const out = template.replace(/\{(title|text)\}/g, (_, k) => (k === "title" ? t : text));
    return { rendered: hasText ? out : out + text, prefix: hasText ? null : template.replace(/\{title\}/g, t) };
  };

  // ---- 通知 ----------------------------------------------------------------
  EB.toast = function (message, type = "info", ms = 4000) {
    let box = document.getElementById("toasts");
    if (!box) {
      box = EB.h("div", { id: "toasts" });
      document.body.appendChild(box);
    }
    const t = EB.h("div", { class: "toast " + (type === "error" ? "error" : ""), text: message });
    box.appendChild(t);
    setTimeout(() => t.remove(), type === "error" ? Math.max(ms, 8000) : ms);
  };

  EB.guard = function (fn) {
    return async function (...args) {
      try {
        return await fn.apply(this, args);
      } catch (e) {
        console.error(e);
        EB.toast(e.message || String(e), "error");
      }
    };
  };

  EB.busy = async function (button, fn) {
    const old = button.textContent;
    button.disabled = true;
    button.textContent = "處理中…";
    try {
      return await fn();
    } catch (e) {
      console.error(e);
      EB.toast(e.message || String(e), "error");
    } finally {
      button.disabled = false;
      button.textContent = old;
    }
  };

  /* 簡單的輸入對話框，回傳輸入值（取消回傳 null）。 */
  EB.promptDialog = function (title, label, initial = "", help = "") {
    return new Promise((resolve) => {
      const input = EB.h("input", { type: "text", value: initial });
      const dlg = EB.h(
        "dialog",
        {},
        EB.h("h2", { text: title }),
        EB.h("label", { class: "field" }, EB.h("span", { class: "field-label", text: label }), input, help ? EB.h("span", { class: "field-help", text: help }) : null),
        EB.h(
          "div",
          { class: "row end", style: { marginTop: "16px" } },
          EB.h("button", { type: "button", text: "取消", onclick: () => close(null) }),
          EB.h("button", { type: "button", class: "primary", text: "確定", onclick: () => close(input.value.trim()) })
        )
      );
      function close(v) {
        dlg.close();
        dlg.remove();
        resolve(v);
      }
      input.addEventListener("keydown", (e) => {
        if (e.key === "Enter") close(input.value.trim());
      });
      dlg.addEventListener("cancel", () => close(null));
      document.body.appendChild(dlg);
      dlg.showModal();
      input.focus();
      input.select();
    });
  };

  EB.textDialog = function (title, text) {
    const pre = EB.h("pre", { class: "preview", style: { maxHeight: "60vh", overflow: "auto" }, text });
    const dlg = EB.h(
      "dialog",
      {},
      EB.h("h2", { text: title }),
      pre,
      EB.h(
        "div",
        { class: "row end", style: { marginTop: "16px" } },
        EB.h("button", {
          type: "button",
          text: "複製",
          onclick: async () => {
            try {
              await navigator.clipboard.writeText(text);
              EB.toast("已複製");
            } catch {
              EB.toast("無法存取剪貼簿，請手動選取複製", "error");
            }
          },
        }),
        EB.h("button", { type: "button", class: "primary", text: "關閉", onclick: () => dlg.close() })
      )
    );
    dlg.addEventListener("close", () => dlg.remove());
    document.body.appendChild(dlg);
    dlg.showModal();
  };

  // ---- 參數表單（依 provider 的 params 定義自動產生） -------------------------
  EB.paramForm = function (container, schema, values) {
    EB.clear(container);
    values = values || {};
    const basic = EB.h("div", { class: "params" });
    const advGrid = EB.h("div", { class: "params" });
    const adv = EB.h("details", { class: "advanced" }, EB.h("summary", { text: "進階參數" }), advGrid);
    const inputs = {};
    for (const p of schema.params) {
      const value = values[p.name] !== undefined ? values[p.name] : p.default;
      let input;
      if (p.kind === "bool") {
        input = EB.h("input", { type: "checkbox", checked: !!value });
      } else if (p.kind === "select") {
        input = EB.h("select", {}, (p.options || []).map((o) => EB.h("option", { value: o, text: o, selected: o === value })));
      } else if (p.kind === "json") {
        const text = value && Object.keys(value).length ? JSON.stringify(value) : "";
        input = EB.h("textarea", { rows: 2, placeholder: "{}", spellcheck: "false" });
        input.value = text;
      } else {
        input = EB.h("input", {
          type: p.kind === "int" || p.kind === "float" ? "number" : "text",
          step: p.kind === "float" ? "any" : null,
          placeholder: p.placeholder || "",
          spellcheck: "false",
        });
        input.value = value === null || value === undefined ? "" : value;
      }
      input.dataset.param = p.name;
      inputs[p.name] = input;
      let field;
      if (p.kind === "bool") {
        field = EB.h(
          "label",
          { class: "field" },
          EB.h("span", { class: "check" }, input, EB.h("span", { class: "field-label", style: { margin: 0 }, text: p.label })),
          p.help ? EB.h("span", { class: "field-help", text: p.help }) : null
        );
      } else {
        const status = p.name.endsWith("_env") ? EB.h("span", { class: "env-status" }) : null;
        field = EB.h(
          "label",
          { class: "field" + (p.kind === "json" || p.name === "base_url" ? " wide" : "") },
          EB.h("span", { class: "field-label" }, p.label, " ", status),
          input,
          p.help ? EB.h("span", { class: "field-help", text: p.help }) : null
        );
        if (status) {
          const check = async () => {
            const name = input.value.trim();
            EB.clear(status);
            if (!name) return;
            try {
              const env = await EB.api("/api/env?name=" + encodeURIComponent(name));
              status.appendChild(
                env[name] ? EB.h("span", { class: "chip good", text: "已設定" }) : EB.h("span", { class: "chip bad", text: "未設定" })
              );
            } catch {
              /* 忽略 */
            }
          };
          input.addEventListener("change", check);
          check();
        }
      }
      (p.advanced ? advGrid : basic).appendChild(field);
    }
    container.appendChild(basic);
    if (advGrid.children.length) container.appendChild(adv);

    return {
      inputs,
      values() {
        const out = {};
        for (const p of schema.params) {
          const input = inputs[p.name];
          if (p.kind === "bool") out[p.name] = input.checked;
          else if (p.kind === "json") {
            const t = input.value.trim();
            if (!t) out[p.name] = {};
            else {
              try {
                out[p.name] = JSON.parse(t);
              } catch {
                throw new Error(`參數「${p.label}」不是合法的 JSON`);
              }
            }
          } else if (p.kind === "int" || p.kind === "float") {
            const t = input.value.trim();
            out[p.name] = t === "" ? null : Number(t);
          } else out[p.name] = input.value.trim();
        }
        return out;
      },
      set(name, value) {
        const input = inputs[name];
        if (!input) return;
        if (input.type === "checkbox") input.checked = !!value;
        else input.value = value === null || value === undefined ? "" : value;
        input.dispatchEvent(new Event("input", { bubbles: true }));
      },
    };
  };

  // ---- 主題 ----------------------------------------------------------------
  function storedTheme() {
    try {
      return localStorage.getItem("eb-theme") || "auto";
    } catch {
      return "auto";
    }
  }
  function applyTheme(t) {
    if (t === "auto") delete document.documentElement.dataset.theme;
    else document.documentElement.dataset.theme = t;
    const btn = document.getElementById("theme-toggle");
    if (btn) btn.textContent = { auto: "主題：自動", light: "主題：淺色", dark: "主題：深色" }[t];
  }
  applyTheme(storedTheme());

  document.addEventListener("DOMContentLoaded", () => {
    applyTheme(storedTheme());
    const btn = document.getElementById("theme-toggle");
    if (btn)
      btn.addEventListener("click", () => {
        const next = { auto: "light", light: "dark", dark: "auto" }[storedTheme()];
        try {
          localStorage.setItem("eb-theme", next);
        } catch {
          /* 無法儲存時仍套用 */
        }
        applyTheme(next);
        document.dispatchEvent(new CustomEvent("eb-theme"));
      });

    // 導覽列：執行中任務數
    const badge = document.getElementById("jobs-badge");
    async function poll() {
      try {
        const a = await EB.api("/api/jobs/active");
        const n = a.running + a.queued;
        badge.textContent = n;
        badge.classList.toggle("hidden", n === 0);
        badge.title = `執行中 ${a.running}、排隊中 ${a.queued}`;
      } catch {
        /* 伺服器暫時無回應 */
      }
      setTimeout(poll, 3000);
    }
    if (badge) poll();
  });
})();
