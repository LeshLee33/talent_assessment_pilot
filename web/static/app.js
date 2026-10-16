// Страница оценки кандидата: загрузка исходных данных и расчёт через API.
(function () {
  "use strict";

  const id = window.CANDIDATE_ID;
  const api = "/api/candidates/" + encodeURIComponent(id);
  const $ = (sel) => document.querySelector(sel);
  const btn = $("#evaluate-btn");
  const statusEl = $("#status");
  let candidate = null;
  let scale = { min: 1, max: 5 };

  const FLAG_LABELS = {
    insufficient_data: "недостаточно данных — индикатор не оценён",
    no_self_report: "нет самооценки — учтены только поведенческие методы",
    self_assessment_overestimated: "завышенная самооценка",
    self_assessment_underestimated: "заниженная самооценка",
  };
  const PRIORITY = { high: "Высокий", medium: "Средний", low: "Низкий" };

  function esc(s) {
    return String(s ?? "").replace(/[&<>"']/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }
  const fmt = (x) => (x === null || x === undefined ? "—" : Number(x).toFixed(2));
  const fmtGap = (x) => (x === null || x === undefined ? "—" : (x > 0 ? "+" : "") + Number(x).toFixed(2));
  const pct = (x) => (Math.max(scale.min, Math.min(scale.max, x)) - scale.min) / (scale.max - scale.min) * 100;
  const show = (sel) => $(sel).classList.remove("hidden");

  function setStatus(kind, html) {
    statusEl.className = "notice" + (kind ? " notice--" + kind : "");
    statusEl.innerHTML = html;
    statusEl.classList.remove("hidden");
  }

  async function request(url, options) {
    let res;
    try {
      res = await fetch(url, options);
    } catch (e) {
      throw { message: "Сервер недоступен. Проверьте, что FastAPI запущен.", problems: [] };
    }
    let body = null;
    try { body = await res.json(); } catch (e) { /* не JSON */ }
    if (!res.ok) {
      const d = (body && body.detail) || {};
      const message = typeof d === "string" ? d : d.message || "Ошибка " + res.status;
      const problems = Array.isArray(d.problems) ? d.problems
        : Array.isArray(d) ? d.map((p) => (p.loc || []).join(".") + ": " + p.msg) : [];
      throw { message, problems };
    }
    return body;
  }

  function errorHtml(err) {
    let html = "<strong>" + esc(err.message || "Неизвестная ошибка") + "</strong>";
    if (err.problems && err.problems.length) {
      html += "<ul>" + err.problems.map((p) => "<li>" + esc(p) + "</li>").join("") + "</ul>";
    }
    return html;
  }

  function bar(level, target, cat) {
    const fill = level === null || level === undefined ? 0 : pct(level);
    return '<div class="bar bar--' + esc(cat) + '" title="Уровень ' + fmt(level) + ", цель " + fmt(target) + '">' +
      '<div class="bar__fill" style="width:' + fill + '%"></div>' +
      '<div class="bar__target" style="left:' + pct(target) + '%"></div></div>';
  }

  function indicatorsTable(indicators, methods) {
    const rows = indicators.map((ind) => {
      const scores = Object.entries(ind.scores)
        .map(([m, v]) => esc(methods[m] || m) + ": <strong>" + (v === null ? "нет данных" : esc(v)) + "</strong>")
        .join("<br>");
      return "<tr><td>" + esc(ind.id) + "</td><td>" + esc(ind.name) + "</td><td>" + scores + "</td>" +
        '<td class="num">' + fmt(ind.level) + "</td><td>" +
        (ind.flags || []).map((f) => esc(FLAG_LABELS[f] || f)).join("; ") + "</td></tr>";
    }).join("");
    return '<table class="ind-table"><thead><tr><th>ID</th><th>Индикатор</th><th>Баллы по методам</th>' +
      '<th class="num">Балл</th><th>Флаги</th></tr></thead><tbody>' + rows + "</tbody></table>";
  }

  // Таблица компетенций: до расчёта — только целевые уровни, после — полный результат.
  function renderCompetencies(report) {
    const resById = {};
    if (report) report.competencies.forEach((c) => { resById[c.id] = c; });
    const methods = (report || candidate).methods;

    $("#comp-body").innerHTML = candidate.competencies.map((comp) => {
      const r = resById[comp.id];
      const cat = r ? r.category.id : "not_assessed";
      const gapCls = r && r.gap !== null ? (r.gap >= 0 ? "gap--pos" : "gap--neg") : "";
      const indicators = comp.indicators.map((ind) => {
        const ri = r ? r.indicators[ind.id] : null;
        return { id: ind.id, name: ind.name, scores: ind.scores, level: ri ? ri.level : null, flags: ri ? ri.flags : [] };
      });
      return '<tr class="comp-row row--' + esc(cat) + '" data-comp="' + esc(comp.id) + '" tabindex="0" aria-expanded="false">' +
        '<td><span class="chev">▸</span>' + esc(comp.name) + "</td>" +
        '<td class="num"><strong>' + (r ? fmt(r.level) : "—") + "</strong></td>" +
        '<td class="num">' + Number(comp.target).toFixed(1) + "</td>" +
        '<td class="num ' + gapCls + '">' + (r ? fmtGap(r.gap) : "—") + "</td>" +
        '<td class="bar-col">' + bar(r ? r.level : null, comp.target, cat) + "</td>" +
        "<td>" + (r ? '<span class="tag tag--' + esc(cat) + '">' + esc(r.category.name) + "</span>"
          : '<span class="muted small">не рассчитано</span>') + "</td></tr>" +
        '<tr class="ind-row hidden" data-ind="' + esc(comp.id) + '"><td colspan="6">' +
        indicatorsTable(indicators, methods) + "</td></tr>";
    }).join("");
  }

  function toggleRow(row) {
    const detail = document.querySelector('tr[data-ind="' + row.dataset.comp + '"]');
    const open = detail.classList.toggle("hidden") === false;
    row.setAttribute("aria-expanded", String(open));
    row.querySelector(".chev").textContent = open ? "▾" : "▸";
  }

  function listItems(items, empty) {
    if (!items.length) return '<li class="empty">' + esc(empty) + "</li>";
    return items.map((c) => "<li><span>" + esc(c.name) + "</span>" +
      '<span class="tag tag--' + esc(c.category.id) + '">' + fmt(c.level) + " / " + Number(c.target).toFixed(1) +
      " (" + fmtGap(c.gap) + ")</span></li>").join("");
  }

  function renderReport(r) {
    scale = r.scale;
    renderCompetencies(r);
    const o = r.overall;
    $("#overall").innerHTML =
      '<div class="overall__verdict-wrap"><div class="stat__label">Итог</div>' +
      '<div class="overall__verdict overall__verdict--' + esc(o.id) + '">' + esc(o.name) + "</div>" +
      (o.capped_by_critical ? '<div class="muted small">Ограничено из-за критичной зоны развития</div>' : "") + "</div>" +
      '<div><div class="stat__label">Средний уровень</div><div class="stat__value">' + fmt(o.average_level) + "</div></div>" +
      '<div><div class="stat__label">Средняя цель</div><div class="stat__value">' + fmt(o.average_target) + "</div></div>" +
      '<div><div class="stat__label">Разница</div><div class="stat__value ' +
        (o.average_gap >= 0 ? "gap--pos" : "gap--neg") + '">' + fmtGap(o.average_gap) + "</div></div>" +
      '<div><div class="stat__label">На уровне цели</div><div class="stat__value">' + o.matched + " из " + o.assessed + "</div>" +
      (o.total > o.assessed ? '<div class="muted small">не оценено: ' + (o.total - o.assessed) + "</div>" : "") + "</div>";
    $("#generated-at").textContent = "Рассчитано: " + new Date(r.meta.generated_at).toLocaleString("ru-RU");

    $("#strengths").innerHTML = listItems(r.strengths, "Компетенций выше цели на 0.5 и более нет");
    $("#meets").innerHTML = listItems(r.meets, "Нет");
    $("#development").innerHTML = listItems(r.development_areas.concat(r.not_assessed), "Зон развития не выявлено");

    $("#recommendations").innerHTML = r.recommendations.map((rec) =>
      '<li><span class="prio prio--' + esc(rec.priority) + '">' + esc(PRIORITY[rec.priority] || rec.priority) +
      "</span>" + esc(rec.text) + "</li>").join("");

    if (r.flags.length) {
      $("#flags").innerHTML = r.flags.map((f) => "<li><span>" + esc(f.indicator) + " · " + esc(f.name) + "</span>" +
        '<span class="muted small">' + f.flags.map((x) => esc(FLAG_LABELS[x] || x)).join("; ") + "</span></li>").join("");
      show("#flags-card");
    } else {
      $("#flags-card").classList.add("hidden");
    }
    ["#overall-card", "#summary-grid", "#recs-card"].forEach(show);
  }

  async function evaluate() {
    btn.disabled = true;
    setStatus("", '<span class="spinner"></span>Расчёт оценки…');
    try {
      const report = await request(api + "/evaluate", { method: "POST" });
      renderReport(report);
      setStatus("ok", "Оценка рассчитана по правилам из config/. " + esc(report.meta.disclaimer));
    } catch (err) {
      setStatus("error", errorHtml(err));
    } finally {
      btn.disabled = false;
    }
  }

  async function load() {
    try {
      candidate = await request(api);
      scale = candidate.scale;
      $("#profile-note").textContent = candidate.profile_note;
      renderCompetencies(null);
      show("#competencies-card");
      btn.disabled = false;
      setStatus("", "Данные загружены. Нажмите «Рассчитать оценку», чтобы получить результат.");
    } catch (err) {
      setStatus("error", errorHtml(err));
    }
  }

  btn.addEventListener("click", evaluate);
  $("#comp-body").addEventListener("click", (e) => {
    const row = e.target.closest("tr.comp-row");
    if (row) toggleRow(row);
  });
  $("#comp-body").addEventListener("keydown", (e) => {
    const row = e.target.closest("tr.comp-row");
    if (row && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); toggleRow(row); }
  });
  load();
})();
