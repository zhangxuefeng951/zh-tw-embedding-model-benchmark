/* 圖表：水平長條圖（排行榜）、折線圖（K 值曲線）。
 * 規範：細長條（≤24px）、資料端 4px 圓角、基線為 0、1px 實線格線、2px 線條、≥8px 端點加 2px 表面色外環、
 * 每個標記都有 hover / focus 提示、文字一律使用文字色（不用資料色）。 */
(function () {
  "use strict";
  const { h, clear } = window.EB;

  function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  function tooltip(container) {
    let tt = container.querySelector(".tooltip");
    if (!tt) {
      tt = h("div", { class: "tooltip hidden", role: "status" });
      container.appendChild(tt);
    }
    return {
      show(x, y, content) {
        clear(tt);
        EB.append(tt, [content]);
        tt.classList.remove("hidden");
        const cw = container.clientWidth;
        const tw = tt.offsetWidth;
        tt.style.left = Math.min(Math.max(0, x + 12), cw - tw) + "px";
        tt.style.top = Math.max(0, y - tt.offsetHeight - 10) + "px";
      },
      hide() {
        tt.classList.add("hidden");
      },
    };
  }

  /* 水平長條圖。items: [{label, sub, value, muted, title}]，數值範圍固定 0～1（指標都是比例）。 */
  EB.barChart = function (container, items, opts = {}) {
    clear(container);
    container.classList.add("chart");
    if (!items.length) {
      container.appendChild(h("div", { class: "empty", text: opts.emptyText || "沒有資料" }));
      return;
    }
    const max = opts.max || 1;
    const labelWidth = opts.labelWidth || "minmax(160px, 34%)";
    const ticks = [0, 0.25, 0.5, 0.75, 1].map((t) => t * max);
    const grid = h("div", {
      style: { display: "grid", gridTemplateColumns: `${labelWidth} 1fr 64px`, columnGap: "12px", rowGap: "6px", alignItems: "center" },
    });
    const tt = tooltip(container);

    // 刻度列
    grid.appendChild(h("div"));
    const axis = h("div", { style: { position: "relative", height: "16px" } });
    ticks.forEach((t) => {
      axis.appendChild(
        h("span", {
          class: "axis-text small",
          style: { position: "absolute", left: `${(t / max) * 100}%`, transform: "translateX(-50%)", color: "var(--muted)", fontSize: "12px" },
          text: t.toFixed(2).replace(/\.?0+$/, "") || "0",
        })
      );
    });
    grid.appendChild(axis);
    grid.appendChild(h("div"));

    items.forEach((item, i) => {
      const label = h(
        "div",
        { style: { minWidth: 0 } },
        h("div", { style: { whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis", fontSize: "13.5px" }, title: item.title || item.label, text: item.label }),
        item.sub ? h("div", { style: { whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis", fontSize: "12px", color: "var(--muted)" }, text: item.sub }) : null
      );
      const track = h("div", { style: { position: "relative", height: "28px" } });
      ticks.forEach((t) => {
        track.appendChild(
          h("div", {
            style: {
              position: "absolute",
              left: `${(t / max) * 100}%`,
              top: 0,
              bottom: 0,
              width: "1px",
              background: t === 0 ? "var(--axis)" : "var(--grid)",
            },
          })
        );
      });
      const v = Math.max(0, Math.min(max, item.value || 0));
      const bar = h("div", {
        class: "bar",
        style: {
          position: "absolute",
          left: 0,
          top: "6px",
          height: "16px",
          width: `calc(${(v / max) * 100}% )`,
          minWidth: v > 0 ? "2px" : "0",
          background: item.muted ? "var(--baseline-bar)" : "var(--series-1)",
          borderRadius: "0 4px 4px 0",
        },
      });
      track.appendChild(bar);
      const value = h("div", { class: "num", style: { fontSize: "13px", color: "var(--text)" }, text: EB.fmt(item.value) });
      const row = [label, track, value];
      const showTip = (ev) => {
        const rect = container.getBoundingClientRect();
        const tr = track.getBoundingClientRect();
        const x = ev && ev.clientX ? ev.clientX - rect.left : tr.left - rect.left + tr.width * (v / max);
        const y = tr.top - rect.top + 4;
        bar.style.opacity = "0.8";
        tt.show(
          x,
          y,
          h(
            "div",
            {},
            h("div", { class: "tt-title", text: item.title || item.label }),
            h("div", { class: "tt-row" }, h("strong", { text: EB.fmt(item.value) }), h("span", { class: "secondary", text: opts.metricLabel || "" })),
            item.sub ? h("div", { class: "muted small", text: item.sub }) : null
          )
        );
      };
      const hideTip = () => {
        bar.style.opacity = "";
        tt.hide();
      };
      track.tabIndex = 0;
      track.setAttribute("aria-label", `${item.label}：${EB.fmt(item.value)}`);
      for (const el of row) {
        el.addEventListener("pointermove", showTip);
        el.addEventListener("pointerleave", hideTip);
      }
      track.addEventListener("focus", () => showTip(null));
      track.addEventListener("blur", hideTip);
      if (item.onclick) {
        label.style.cursor = "pointer";
        label.addEventListener("click", item.onclick);
      }
      row.forEach((el) => grid.appendChild(el));
      if (i === 0) row.forEach((el) => el.setAttribute("data-first", ""));
    });
    container.appendChild(grid);
  };

  /* 折線圖。data: {xs: [...], series: [{name, values, color}]}，y 範圍 0～1。 */
  EB.lineChart = function (container, data, opts = {}) {
    clear(container);
    container.classList.add("chart");
    const series = data.series.filter((s) => s.values && s.values.length);
    if (!series.length) return;
    // 圖例（≥ 2 個序列時一定要有）
    if (series.length > 1) {
      container.appendChild(
        h(
          "div",
          { class: "legend" },
          series.map((s) => h("span", {}, h("span", { class: "key", style: { background: s.color } }), s.name))
        )
      );
    }
    const width = Math.max(320, container.clientWidth || 600);
    const height = opts.height || 260;
    // 右側留給末端直接標籤：依實際文字寬度計算
    const ctx2d = document.createElement("canvas").getContext("2d");
    ctx2d.font = `12px ${getComputedStyle(document.body).fontFamily}`;
    const labelW = Math.max(...series.map((ser) => ctx2d.measureText(`${ser.name} 0.000`).width));
    const m = { top: 12, right: Math.min(Math.ceil(labelW) + 20, width * 0.4), bottom: 28, left: 40 };
    const iw = width - m.left - m.right;
    const ih = height - m.top - m.bottom;
    const xs = data.xs;
    const xMin = xs[0];
    const xMax = xs[xs.length - 1];
    const x = (v) => m.left + (xMax === xMin ? iw / 2 : ((v - xMin) / (xMax - xMin)) * iw);
    const y = (v) => m.top + (1 - Math.max(0, Math.min(1, v))) * ih;
    const NS = "http://www.w3.org/2000/svg";
    const s = (tag, attrs, ...children) => {
      const el = document.createElementNS(NS, tag);
      for (const [k, v] of Object.entries(attrs || {})) if (v !== null && v !== undefined) el.setAttribute(k, v);
      for (const c of children) if (c) el.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
      return el;
    };
    const svg = s("svg", { viewBox: `0 0 ${width} ${height}`, height, role: "img", "aria-label": opts.ariaLabel || "折線圖", tabindex: "0" });
    // 格線與刻度
    [0, 0.25, 0.5, 0.75, 1].forEach((t) => {
      svg.appendChild(s("line", { class: t === 0 ? "baseline" : "gridline", x1: m.left, x2: m.left + iw, y1: y(t), y2: y(t) }));
      svg.appendChild(s("text", { class: "axis-text", x: m.left - 8, y: y(t) + 4, "text-anchor": "end" }, t.toFixed(2)));
    });
    const xticks = opts.xticks || xs;
    xticks.forEach((t) => svg.appendChild(s("text", { class: "axis-text", x: x(t), y: m.top + ih + 18, "text-anchor": "middle" }, String(t))));
    if (opts.xLabel) svg.appendChild(s("text", { class: "axis-text", x: m.left + iw, y: m.top + ih + 18, "text-anchor": "start", dx: 8 }, opts.xLabel));

    const surface = cssVar("--surface") || "#fff";
    series.forEach((ser) => {
      const d = ser.values.map((v, i) => `${i ? "L" : "M"}${x(xs[i]).toFixed(1)},${y(v).toFixed(1)}`).join("");
      svg.appendChild(s("path", { d, fill: "none", stroke: ser.color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }));
      const last = ser.values.length - 1;
      svg.appendChild(s("circle", { cx: x(xs[last]), cy: y(ser.values[last]), r: 4, fill: ser.color, stroke: surface, "stroke-width": 2 }));
    });
    // 末端直接標籤：彼此太近時不硬擠，改由圖例與提示框承擔
    const ends = series.map((ser) => ({ ser, y: y(ser.values[ser.values.length - 1]) })).sort((a, b) => a.y - b.y);
    const separated = ends.every((e, i) => i === 0 || e.y - ends[i - 1].y >= 15);
    if (separated) {
      ends.forEach((e) => {
        const v = e.ser.values[e.ser.values.length - 1];
        svg.appendChild(s("text", { x: m.left + iw + 10, y: e.y + 4, fill: cssVar("--text-2") }, `${e.ser.name} ${EB.fmt(v, 3)}`));
      });
    }
    // 十字線 + 提示框
    const cross = s("line", { class: "baseline", y1: m.top, y2: m.top + ih, visibility: "hidden" });
    svg.appendChild(cross);
    const dots = series.map((ser) => {
      const c = s("circle", { r: 4, fill: ser.color, stroke: surface, "stroke-width": 2, visibility: "hidden" });
      svg.appendChild(c);
      return c;
    });
    const overlay = s("rect", { x: m.left, y: m.top, width: iw, height: ih, fill: "transparent" });
    svg.appendChild(overlay);
    container.appendChild(svg);
    const tt = tooltip(container);
    let current = -1;
    function showAt(i) {
      current = i;
      const px = x(xs[i]);
      cross.setAttribute("x1", px);
      cross.setAttribute("x2", px);
      cross.setAttribute("visibility", "visible");
      series.forEach((ser, j) => {
        dots[j].setAttribute("cx", px);
        dots[j].setAttribute("cy", y(ser.values[i]));
        dots[j].setAttribute("visibility", "visible");
      });
      const scale = svg.getBoundingClientRect().width / width;
      const rows = series
        .map((ser) => ({ ser, v: ser.values[i] }))
        .sort((a, b) => b.v - a.v)
        .map(({ ser, v }) => h("div", { class: "tt-row" }, h("span", { class: "key", style: { background: ser.color } }), h("strong", { text: EB.fmt(v) }), h("span", { class: "secondary", text: ser.name })));
      tt.show(px * scale, m.top * scale + 20, h("div", {}, h("div", { class: "tt-title", text: `${opts.xName || "K"} = ${xs[i]}` }), rows));
    }
    function hide() {
      current = -1;
      cross.setAttribute("visibility", "hidden");
      dots.forEach((d) => d.setAttribute("visibility", "hidden"));
      tt.hide();
    }
    overlay.addEventListener("pointermove", (ev) => {
      const rect = svg.getBoundingClientRect();
      const px = ((ev.clientX - rect.left) / rect.width) * width;
      let best = 0;
      xs.forEach((v, i) => {
        if (Math.abs(x(v) - px) < Math.abs(x(xs[best]) - px)) best = i;
      });
      showAt(best);
    });
    overlay.addEventListener("pointerleave", hide);
    svg.addEventListener("keydown", (ev) => {
      if (ev.key === "ArrowRight") showAt(Math.min(xs.length - 1, current + 1));
      else if (ev.key === "ArrowLeft") showAt(Math.max(0, current < 0 ? 0 : current - 1));
      else if (ev.key === "Escape") hide();
    });
    svg.addEventListener("blur", hide);
  };
})();
