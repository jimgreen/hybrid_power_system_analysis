const state = {
  schemes: [],
  currentScheme: "",
  optimization: null,
  pollTimer: null,
  pollDelay: 4000,
  optimizationCommandHeight: null,
  optimizationResultHeight: null,
  optimizationLogHeight: null,
};

const optimizationResizeMinHeights = {
  command: 112,
  result: 220,
  log: 120,
};

const resultTabLabels = {
  overview: "结果概览",
  green: "绿电结果",
  safety: "安全结果",
};

const greenDailySeries = [
  { key: "diesel_energy", label: "柴发日电量", direction: "up", color: "#7a6650" },
  { key: "wind_energy", label: "风电日电量", direction: "up", color: "#2a9d8f" },
  { key: "pv_energy", label: "光伏日电量", direction: "up", color: "#d8a31a" },
  { key: "hydrogen_energy", label: "氢能日电量", direction: "up", color: "#4d7fd1" },
  { key: "load_energy", label: "负荷电量", direction: "down", color: "#c7504a" },
  { key: "hydrogen_production_energy", label: "制氢电量", direction: "down", color: "#6b5fb5" },
];

document.addEventListener("DOMContentLoaded", () => {
  bindResultTabs();
  bindOptimizationActions();
  lockOptimizationCommandHeight();
  bindOptimizationResultResizeHandle();
  bindOptimizationLogResizeHandle();
  window.addEventListener("resize", () => {
    state.optimizationCommandHeight = null;
    lockOptimizationCommandHeight();
  });
  loadSchemes().then(() => refreshOptimizationStatus()).catch(showError);
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
      state.optimization = defaultOptimizationState(state.currentScheme);
      renderOptimization(state.optimization);
      refreshOptimizationStatus(state.currentScheme).catch(showError);
    });
  });
}

function renderCurrentScheme() {
  const current = document.getElementById("optimizationCurrentScheme");
  current.textContent = `当前方案: ${state.currentScheme || "未选择方案"}`;
  window.requestAnimationFrame(lockOptimizationCommandHeight);
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

async function refreshOptimizationStatus(scheme = state.currentScheme) {
  const data = await api(optimizationStatusPath(scheme));
  if (scheme !== state.currentScheme) return;
  state.optimization = data;
  renderOptimization(data);
  scheduleOptimizationPolling();
}

function optimizationStatusPath(scheme) {
  return scheme ? `/api/optimization/status?scheme=${encodeURIComponent(scheme)}` : "/api/optimization/status";
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
  updateOptimizationActions(data);
  renderMetrics(data.metrics || []);
  renderOverviewTables(data.results?.overview_tables || defaultOverviewTables(), data.results?.overview_disks || defaultOverviewDisks());
  renderGreenResult(data.results?.green_table || defaultGreenTable(), data.results?.curves?.green_daily || []);
  renderResultPanel("safety", resultTabLabels.safety, data.results?.safety || [], data.results?.curves?.safety || []);
  renderOptimizationLogs(data.logs || []);
  window.requestAnimationFrame(lockOptimizationCommandHeight);
}

function updateOptimizationActions(data = state.optimization || {}) {
  const startButton = document.getElementById("startOptimization");
  const stopButton = document.getElementById("stopOptimization");
  if (!startButton || !stopButton) return;
  const hasScheme = Boolean(state.currentScheme);
  const isRunning = data.status === "运行中";
  startButton.disabled = !hasScheme || isRunning;
  stopButton.disabled = !hasScheme || !isRunning;
  startButton.classList.toggle("is-disabled", startButton.disabled);
  stopButton.classList.toggle("is-disabled", stopButton.disabled);
  startButton.classList.toggle("is-active", !startButton.disabled);
  stopButton.classList.toggle("is-active", !stopButton.disabled);
  startButton.setAttribute("aria-disabled", String(startButton.disabled));
  stopButton.setAttribute("aria-disabled", String(stopButton.disabled));
  startButton.title = !hasScheme ? "请先选择方案" : isRunning ? "当前方案正在运行" : "启动当前方案优化规划";
  stopButton.title = !hasScheme ? "请先选择方案" : isRunning ? "停止当前方案优化规划" : "当前方案没有运行";
}

function defaultOptimizationState(scheme = "") {
  return {
    status: "待启动",
    scheme,
    start_time: "",
    end_time: "",
    progress: 0,
    metrics: [
      { label: "当前状态", value: "待启动", unit: "" },
      { label: "启动时刻", value: "-", unit: "" },
      { label: "结束时刻", value: "-", unit: "" },
      { label: "度电成本", value: "-", unit: "元/kWh" },
      { label: "绿电占比", value: "-", unit: "%" },
    ],
    results: {
      overview_tables: defaultOverviewTables(),
      overview_disks: defaultOverviewDisks(),
      green: [],
      green_table: defaultGreenTable(),
      safety: [],
      curves: { green: [], green_daily: [], safety: [] },
    },
    logs: [{ time: "", level: "", message: "正在加载当前方案优化状态" }],
    running_schemes: [],
  };
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

function defaultOverviewTables() {
  return [
    { title: "规划结果", rows: [{ "设备类型": "-", "设计台数": "-", "单台容量": "-", "总容量": "-", "单位": "" }] },
    { title: "规划年指标", rows: [{ "指标": "-", "数值": "-", "单位": "" }] },
  ];
}

function defaultOverviewDisks() {
  return [
    { title: "成本构成", left_label: "运行成本", left_value: 0, right_label: "建设成本", right_value: 0, unit: "万元" },
    { title: "电量构成", left_label: "柴发电量", left_value: 0, right_label: "新能源电量", right_value: 0, unit: "MWh" },
  ];
}

function defaultGreenTable() {
  return [
    { "指标": "负荷总电量(kWh)", "数值": "-" },
    { "指标": "柴发总电量(kWh)", "数值": "-" },
    { "指标": "风机总发电量(kWh)", "数值": "-" },
    { "指标": "光伏总发电量(kWh)", "数值": "-" },
    { "指标": "电储总发电量(kWh)", "数值": "-" },
    { "指标": "氢储总发电量(kWh)", "数值": "-" },
    { "指标": "新能源总弃电量(%)", "数值": "-" },
    { "指标": "柴油消耗(吨)", "数值": "-" },
    { "指标": "制氢总量(Nm3)", "数值": "-" },
  ];
}

function renderOverviewTables(tables, disks) {
  const panel = document.getElementById("overviewResult");
  if (!panel) return;
  const safeTables = tables.length ? tables : defaultOverviewTables();
  panel.innerHTML = `
    <div class="optimization-overview-grid">
      ${renderOverviewTableCard(safeTables[0] || defaultOverviewTables()[0])}
      ${renderOverviewDisks(disks?.length ? disks : defaultOverviewDisks())}
      ${renderOverviewTableCard(safeTables[1] || defaultOverviewTables()[1])}
    </div>`;
}

function renderOverviewTableCard(table) {
  return `
    <section class="overview-table-card">
      <h2>${escapeHtml(table.title || "")}</h2>
      <div class="data-table optimization-overview-table">${renderResultTable(table.rows || [])}</div>
    </section>`;
}

function renderOverviewDisks(disks) {
  return `<section class="overview-ratio-stack">${disks.map(renderOverviewDisk).join("")}</section>`;
}

function renderOverviewDisk(disk) {
  const leftValue = Number(disk.left_value) || 0;
  const rightValue = Number(disk.right_value) || 0;
  const total = Math.max(leftValue + rightValue, 0.0001);
  const percent = Math.round((leftValue / total) * 100);
  return `
    <div class="ratio-disk-card">
      <div class="ratio-disk" style="--ratio-percent:${percent}">
        <span>${percent}%</span>
      </div>
      <div class="ratio-disk-info">
        <h2>${escapeHtml(disk.title || "")}</h2>
        <div class="ratio-legend">
          <div><span class="legend-dot primary-dot"></span><span>${escapeHtml(disk.left_label || "")}</span><strong>${escapeHtml(formatNumber(leftValue))}${escapeHtml(disk.unit || "")}</strong></div>
          <div><span class="legend-dot secondary-dot"></span><span>${escapeHtml(disk.right_label || "")}</span><strong>${escapeHtml(formatNumber(rightValue))}${escapeHtml(disk.unit || "")}</strong></div>
        </div>
      </div>
    </div>`;
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

function renderGreenResult(rows, dailyPoints) {
  const panel = document.getElementById("greenResult");
  if (!panel) return;
  const safeRows = rows.length ? rows : defaultGreenTable();
  const formattedRows = safeRows.map((row) => ({
    "指标": row["指标"],
    "数值": typeof row["数值"] === "number" ? formatNumber(row["数值"]) : row["数值"],
  }));
  panel.innerHTML = `
    <div class="green-result-layout">
      <div class="data-table green-result-table">${renderResultTable(formattedRows)}</div>
      <section class="green-chart-card green-daily-chart" aria-label="${escapeHtml(resultTabLabels.green)}日曲线">
        ${renderGreenDailyChart(dailyPoints)}
      </section>
    </div>`;
}

function renderGreenDailyChart(points) {
  if (!points.length) return '<div class="empty-summary">暂无日曲线</div>';
  const width = 1000;
  const height = 330;
  const margin = { top: 18, right: 22, bottom: 38, left: 72 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const upSeries = greenDailySeries.filter((series) => series.direction === "up");
  const downSeries = greenDailySeries.filter((series) => series.direction === "down");
  const upMax = Math.max(
    ...points.map((point) => upSeries.reduce((total, series) => total + numericValue(point[series.key]), 0)),
    1,
  );
  const downMax = Math.max(
    ...points.map((point) => downSeries.reduce((total, series) => total + numericValue(point[series.key]), 0)),
    1,
  );
  const zeroY = margin.top + plotHeight * (upMax / (upMax + downMax));
  const topSpan = Math.max(1, zeroY - margin.top);
  const bottomSpan = Math.max(1, margin.top + plotHeight - zeroY);
  const xAt = (index) => margin.left + (points.length === 1 ? plotWidth / 2 : (index / (points.length - 1)) * plotWidth);
  const yUp = (value) => zeroY - (value / upMax) * topSpan;
  const yDown = (value) => zeroY + (value / downMax) * bottomSpan;
  const positiveTicks = [0.5, 1].map((ratio) => ({
    y: yUp(upMax * ratio),
    label: formatAxisNumber(upMax * ratio),
  }));
  const negativeTicks = [0.5, 1].map((ratio) => ({
    y: yDown(downMax * ratio),
    label: `-${formatAxisNumber(downMax * ratio)}`,
  }));
  return `
    <div class="green-chart-legend">${greenDailySeries
      .map((series) => `<span><i style="background:${series.color}"></i>${escapeHtml(series.label)}</span>`)
      .join("")}</div>
    <svg class="green-chart-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="绿电日曲线">
      <line class="green-axis-line" x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${height - margin.bottom}"></line>
      <line class="green-zero-line" x1="${margin.left}" y1="${zeroY.toFixed(2)}" x2="${width - margin.right}" y2="${zeroY.toFixed(2)}"></line>
      ${positiveTicks.concat(negativeTicks).map((tick) => renderGreenTick(tick, margin.left, width - margin.right)).join("")}
      ${renderGreenStackedAreas(points, upSeries, xAt, yUp)}
      ${renderGreenStackedAreas(points, downSeries, xAt, yDown)}
      ${renderGreenXAxis(points, xAt, zeroY, height - margin.bottom)}
      <text class="green-axis-label" x="14" y="${margin.top + 14}">kWh</text>
      <text class="green-axis-label" x="${width - 76}" y="${height - 8}">日序号</text>
    </svg>`;
}

function renderGreenStackedAreas(points, seriesList, xAt, yAt) {
  const base = points.map(() => 0);
  return seriesList
    .map((series) => {
      const top = points.map((point, index) => base[index] + numericValue(point[series.key]));
      const upperLine = top.map((value, index) => `${xAt(index).toFixed(2)},${yAt(value).toFixed(2)}`).join(" ");
      const lowerLine = base
        .map((value, index) => `${xAt(index).toFixed(2)},${yAt(value).toFixed(2)}`)
        .reverse()
        .join(" ");
      top.forEach((value, index) => {
        base[index] = value;
      });
      return `<polygon class="green-stack-area" points="${upperLine} ${lowerLine}" fill="${series.color}"><title>${escapeHtml(series.label)}</title></polygon>`;
    })
    .join("");
}

function renderGreenTick(tick, left, right) {
  return `
    <line class="green-grid-line" x1="${left}" y1="${tick.y.toFixed(2)}" x2="${right}" y2="${tick.y.toFixed(2)}"></line>
    <text class="green-tick-label" x="${left - 8}" y="${tick.y.toFixed(2)}">${escapeHtml(tick.label)}</text>`;
}

function renderGreenXAxis(points, xAt, zeroY, bottomY) {
  const tickIndexes = [0, 90, 181, 272, points.length - 1].filter((index, position, values) => index >= 0 && index < points.length && values.indexOf(index) === position);
  return tickIndexes
    .map((index) => {
      const x = xAt(index);
      const day = points[index]?.day ?? index + 1;
      return `
        <line class="green-x-tick" x1="${x.toFixed(2)}" y1="${zeroY.toFixed(2)}" x2="${x.toFixed(2)}" y2="${bottomY}"></line>
        <text class="green-x-label" x="${x.toFixed(2)}" y="${bottomY + 24}">${escapeHtml(`第${day}日`)}</text>`;
    })
    .join("");
}

function renderResultTable(rows) {
  if (!rows.length) return '<div class="empty-summary">暂无结果</div>';
  const headers = Object.keys(rows[0]);
  return `<table><thead><tr>${headers.map((header) => `<th>${escapeHtml(header)}</th>`).join("")}</tr></thead><tbody>${rows
    .map((row) => `<tr>${headers.map((header) => `<td>${escapeHtml(row[header])}</td>`).join("")}</tr>`)
    .join("")}</tbody></table>`;
}

function numericValue(value) {
  const number = Number(value);
  return Number.isFinite(number) ? Math.max(0, number) : 0;
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

function formatNumber(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "-";
  return number.toLocaleString("zh-CN", { maximumFractionDigits: 1 });
}

function formatAxisNumber(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "-";
  return Math.round(number).toLocaleString("zh-CN");
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

function bindOptimizationResultResizeHandle() {
  const handle = document.getElementById("optimizationResultResizeHandle");
  const resultCard = document.querySelector(".optimization-result-card");
  if (!handle || !resultCard) return;

  const applyHeight = (height) => {
    const safeHeight = clampOptimizationResultHeight(height);
    const pairedCommandHeight = Math.max(optimizationResizeMinHeights.command, optimizationTopMiddleContentHeight() - safeHeight);
    setOptimizationCommandHeight(pairedCommandHeight);
    setOptimizationResultHeight(safeHeight, handle);
  };

  handle.addEventListener("pointerdown", (event) => {
    event.preventDefault();
    lockOptimizationCommandHeight();
    setOptimizationLogHeight(currentOptimizationLogHeight());
    const startY = event.clientY;
    const startHeight = resultCard.getBoundingClientRect().height || 360;
    handle.classList.add("dragging");
    handle.setPointerCapture?.(event.pointerId);

    const onMove = (moveEvent) => {
      applyHeight(startHeight - (moveEvent.clientY - startY));
    };
    const onDone = () => {
      handle.classList.remove("dragging");
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onDone);
      window.removeEventListener("pointercancel", onDone);
    };

    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onDone);
    window.addEventListener("pointercancel", onDone);
  });

  bindResizeHandleKeys(handle, () => state.optimizationResultHeight || resultCard.getBoundingClientRect().height || 360, applyHeight, optimizationResultHeightBounds);
  handle.setAttribute("aria-valuenow", String(Math.round(resultCard.getBoundingClientRect().height || 360)));
}

function bindOptimizationLogResizeHandle() {
  const handle = document.getElementById("optimizationLogResizeHandle");
  const logCard = document.querySelector(".optimization-log-card");
  if (!handle || !logCard) return;

  const applyHeight = (height) => {
    const safeHeight = clampOptimizationLogHeight(height);
    const pairedResultHeight = Math.max(optimizationResizeMinHeights.result, optimizationResizableContentHeight() - safeHeight);
    setOptimizationLogHeight(safeHeight, handle);
    setOptimizationResultHeight(pairedResultHeight);
  };

  handle.addEventListener("pointerdown", (event) => {
    event.preventDefault();
    lockOptimizationCommandHeight();
    const startY = event.clientY;
    const startHeight = logCard.getBoundingClientRect().height || 180;
    handle.classList.add("dragging");
    handle.setPointerCapture?.(event.pointerId);

    const onMove = (moveEvent) => {
      applyHeight(startHeight - (moveEvent.clientY - startY));
    };
    const onDone = () => {
      handle.classList.remove("dragging");
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onDone);
      window.removeEventListener("pointercancel", onDone);
    };

    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onDone);
    window.addEventListener("pointercancel", onDone);
  });

  bindResizeHandleKeys(handle, () => state.optimizationLogHeight || logCard.getBoundingClientRect().height || 180, applyHeight, optimizationLogHeightBounds);
  handle.setAttribute("aria-valuenow", String(Math.round(logCard.getBoundingClientRect().height || 180)));
}

function bindResizeHandleKeys(handle, currentHeight, applyHeight, boundsFactory) {
  handle.addEventListener("keydown", (event) => {
    const keySteps = {
      ArrowUp: 16,
      ArrowDown: -16,
      PageUp: 64,
      PageDown: -64,
    };
    if (event.key in keySteps) {
      event.preventDefault();
      applyHeight(currentHeight() + keySteps[event.key]);
    } else if (event.key === "Home") {
      event.preventDefault();
      applyHeight(boundsFactory().min);
    } else if (event.key === "End") {
      event.preventDefault();
      applyHeight(boundsFactory().max);
    }
  });
}

function setOptimizationResultHeight(height, handle = document.getElementById("optimizationResultResizeHandle")) {
  const roundedHeight = Math.round(height);
  state.optimizationResultHeight = roundedHeight;
  document.documentElement.style.setProperty("--optimization-result-height", `${roundedHeight}px`);
  handle?.setAttribute("aria-valuenow", String(roundedHeight));
}

function setOptimizationLogHeight(height, handle = document.getElementById("optimizationLogResizeHandle")) {
  const roundedHeight = Math.round(height);
  state.optimizationLogHeight = roundedHeight;
  document.documentElement.style.setProperty("--optimization-log-height", `${roundedHeight}px`);
  handle?.setAttribute("aria-valuenow", String(roundedHeight));
}

function setOptimizationCommandHeight(height) {
  const roundedHeight = Math.round(height);
  state.optimizationCommandHeight = roundedHeight;
  document.documentElement.style.setProperty("--optimization-command-height", `${roundedHeight}px`);
}

function clampOptimizationResultHeight(height) {
  const bounds = optimizationResultHeightBounds();
  return Math.min(Math.max(Number(height) || bounds.min, bounds.min), bounds.max);
}

function clampOptimizationLogHeight(height) {
  const bounds = optimizationLogHeightBounds();
  return Math.min(Math.max(Number(height) || bounds.min, bounds.min), bounds.max);
}

function optimizationResultHeightBounds() {
  const availableHeight = optimizationTopMiddleContentHeight();
  const maxResultHeight = availableHeight - optimizationResizeMinHeights.command;
  return {
    min: optimizationResizeMinHeights.result,
    max: Math.max(optimizationResizeMinHeights.result, Math.min(760, maxResultHeight)),
  };
}

function optimizationLogHeightBounds() {
  const availableHeight = optimizationResizableContentHeight();
  const maxLogHeight = availableHeight - optimizationResizeMinHeights.result;
  return {
    min: optimizationResizeMinHeights.log,
    max: Math.max(optimizationResizeMinHeights.log, Math.min(520, maxLogHeight)),
  };
}

function lockOptimizationCommandHeight() {
  const commandCard = document.querySelector(".optimization-command-card");
  if (!commandCard) return 0;
  const height = Math.ceil(commandCard.getBoundingClientRect().height || commandCard.scrollHeight);
  if (height > 0) {
    setOptimizationCommandHeight(height);
  }
  return height;
}

function optimizationResizableContentHeight() {
  const commandHeight = state.optimizationCommandHeight || lockOptimizationCommandHeight();
  return Math.max(
    optimizationResizeMinHeights.result + optimizationResizeMinHeights.log,
    optimizationCardsContentHeight() - commandHeight,
  );
}

function optimizationTopMiddleContentHeight() {
  const currentLogHeight = currentOptimizationLogHeight();
  return Math.max(
    optimizationResizeMinHeights.command + optimizationResizeMinHeights.result,
    optimizationCardsContentHeight() - currentLogHeight,
  );
}

function optimizationCardsContentHeight() {
  const panel = document.querySelector(".optimization-panel");
  if (!panel) return Math.max(optimizationResizeMinHeights.command + optimizationResizeMinHeights.result + optimizationResizeMinHeights.log, window.innerHeight - 260);
  const style = window.getComputedStyle(panel);
  const paddingY = cssNumber(style.paddingTop) + cssNumber(style.paddingBottom);
  const rowGap = cssNumber(style.rowGap || style.gap);
  const resultHandle = document.getElementById("optimizationResultResizeHandle");
  const logHandle = document.getElementById("optimizationLogResizeHandle");
  const handleHeights =
    (resultHandle?.getBoundingClientRect().height || 14) +
    (logHandle?.getBoundingClientRect().height || 14);
  return Math.max(
    optimizationResizeMinHeights.command + optimizationResizeMinHeights.result + optimizationResizeMinHeights.log,
    panel.clientHeight - paddingY - rowGap * 4 - handleHeights,
  );
}

function currentOptimizationLogHeight() {
  const logCard = document.querySelector(".optimization-log-card");
  return state.optimizationLogHeight || logCard?.getBoundingClientRect().height || 180;
}

function cssNumber(value) {
  const number = Number.parseFloat(value);
  return Number.isFinite(number) ? number : 0;
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[char]);
}

function showError(error) {
  alert(error.message || String(error));
}
