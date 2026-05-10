const state = {
  schemes: [],
  currentScheme: "",
  optimization: null,
  pollTimer: null,
  pollDelay: 4000,
};

document.addEventListener("DOMContentLoaded", () => {
  bindResultTabs();
  bindOptimizationActions();
  loadSchemes().catch(showError);
  refreshOptimizationStatus().catch(showError);
});

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  const data = await response.json();
  if (!response.ok) {
    const error = new Error(data.message || data.error || "请求失败");
    error.payload = data;
    error.status = response.status;
    throw error;
  }
  return data;
}

async function loadSchemes() {
  state.schemes = (await api("/api/planning/schemes")).schemes || [];
  if (!state.currentScheme && state.schemes.length) state.currentScheme = state.schemes[0].name;
  renderSchemes();
  renderCurrentScheme();
}

function renderSchemes() {
  const list = document.getElementById("schemeList");
  if (!state.schemes.length) {
    list.innerHTML = '<div class="validation-item">暂无方案，请先在参数维护中新建方案。</div>';
    return;
  }
  list.innerHTML = state.schemes
    .map((scheme) => `<button class="scheme-item ${scheme.name === state.currentScheme ? "active" : ""}" type="button" data-name="${escapeHtml(scheme.name)}">${escapeHtml(scheme.name)}</button>`)
    .join("");
  list.querySelectorAll(".scheme-item").forEach((button) => {
    button.addEventListener("click", () => {
      state.currentScheme = button.dataset.name || "";
      renderSchemes();
      renderCurrentScheme();
    });
  });
}

function renderCurrentScheme() {
  const current = document.getElementById("optimizationCurrentScheme");
  current.textContent = `当前方案: ${state.currentScheme || "未选择方案"}`;
}

function bindOptimizationActions() {
  document.getElementById("startOptimization").addEventListener("click", () => controlOptimization("start"));
  document.getElementById("stopOptimization").addEventListener("click", () => controlOptimization("stop"));
}

async function controlOptimization(action) {
  if (!state.currentScheme) {
    alert("请先选择方案");
    return;
  }
  try {
    const data = await api("/api/optimization/control", {
      method: "POST",
      body: JSON.stringify({ action, scheme: state.currentScheme }),
    });
    state.optimization = data.state;
    renderOptimization(data.state);
    scheduleOptimizationPolling();
  } catch (error) {
    const data = error.payload || {};
    if (data.error === "running") alert(data.message || "正在运行，无法再次启动");
    else if (data.error === "not_running") alert(data.message || "没有运行");
    else if (data.message) alert(data.message);
    else showError(error);
    await refreshOptimizationStatus().catch(showError);
  }
}

async function refreshOptimizationStatus() {
  const data = await api("/api/optimization/status");
  state.optimization = data;
  renderOptimization(data);
  scheduleOptimizationPolling();
}

function scheduleOptimizationPolling() {
  if (state.pollTimer) window.clearInterval(state.pollTimer);
  const data = state.optimization || {};
  state.pollDelay = data.status === "运行中" ? 1000 : 4000;
  state.pollTimer = window.setInterval(() => {
    refreshOptimizationStatus().catch(showError);
  }, state.pollDelay);
}

function bindResultTabs() {
  const buttons = Array.from(document.querySelectorAll("[data-result-tab]"));
  const panels = Array.from(document.querySelectorAll("[data-result-panel]"));
  buttons.forEach((button) => {
    button.addEventListener("click", () => {
      const target = button.dataset.resultTab;
      buttons.forEach((item) => {
        const active = item === button;
        item.classList.toggle("active", active);
        item.setAttribute("aria-selected", String(active));
      });
      panels.forEach((panel) => {
        const active = panel.dataset.resultPanel === target;
        panel.classList.toggle("active", active);
        panel.hidden = !active;
      });
    });
  });
}

function renderOptimization(data) {
  renderMetrics(data.metrics || []);
  renderResultPanel("overview", "结果概览", data.results?.overview || [], data.results?.curves?.overview || []);
  renderResultPanel("green", "绿电结果", data.results?.green || [], data.results?.curves?.green || []);
  renderResultPanel("safety", "安全结果", data.results?.safety || [], data.results?.curves?.safety || []);
  renderOptimizationLogs(data.logs || []);
}

function renderMetrics(metrics) {
  const byLabel = new Map(metrics.map((item) => [item.label, item]));
  setMetric("optimizationStatus", byLabel.get("当前状态"));
  setMetric("optimizationStartTime", byLabel.get("启动时刻"));
  setMetric("optimizationEndTime", byLabel.get("结束时刻"));
  setMetric("optimizationCost", byLabel.get("度电成本"));
  setMetric("optimizationGreenRatio", byLabel.get("绿电占比"));
}

function setMetric(id, item) {
  const element = document.getElementById(id);
  if (!element) return;
  if (!item) {
    element.textContent = "-";
    return;
  }
  const unit = item.unit ? ` ${item.unit}` : "";
  element.textContent = `${item.value}${unit}`;
}

function renderResultPanel(key, title, rows, points) {
  const panel = document.getElementById(`${key}Result`);
  if (!panel) return;
  panel.innerHTML = `
    <div class="optimization-result-layout">
      <div class="optimization-result-chart" aria-label="${escapeHtml(title)}曲线">${renderMiniBars(points)}</div>
      <div class="data-table optimization-result-table">${renderResultTable(rows)}</div>
    </div>`;
}

function renderResultTable(rows) {
  if (!rows.length) return '<div class="empty-summary">暂无结果</div>';
  const headers = Object.keys(rows[0]);
  return `<table><thead><tr>${headers.map((header) => `<th>${escapeHtml(header)}</th>`).join("")}</tr></thead><tbody>${rows
    .map((row) => `<tr>${headers.map((header) => `<td>${escapeHtml(row[header])}</td>`).join("")}</tr>`)
    .join("")}</tbody></table>`;
}

function renderMiniBars(points) {
  if (!points.length) return '<div class="empty-summary">暂无曲线</div>';
  const maxValue = Math.max(...points.map((point) => Number(point.value) || 0), 1);
  return `<div class="mini-bar-chart">${points
    .map((point) => {
      const value = Number(point.value) || 0;
      const height = Math.max(6, (value / maxValue) * 100);
      return `<div class="mini-bar-item"><div class="mini-bar-value">${escapeHtml(value)}</div><div class="mini-bar-track"><span style="height:${height}%"></span></div><div class="mini-bar-label">${escapeHtml(point.label)}</div></div>`;
    })
    .join("")}</div>`;
}

function renderOptimizationLogs(logs) {
  const box = document.getElementById("optimizationLogs");
  if (!logs.length) {
    box.innerHTML = '<div class="log-line">暂无运行日志</div>';
    return;
  }
  box.innerHTML = logs
    .map((item) => `<div class="log-line ${escapeHtml(item.level || "")}"><span>${escapeHtml(item.time || "")}</span><strong>${escapeHtml(item.message || "")}</strong></div>`)
    .join("");
  box.scrollTop = box.scrollHeight;
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[char]);
}

function showError(error) {
  alert(error.message || String(error));
}
