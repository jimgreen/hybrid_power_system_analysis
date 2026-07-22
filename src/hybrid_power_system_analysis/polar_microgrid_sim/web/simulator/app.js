const apiBase = (window.POLAR_SIM_API_URL || localStorage.getItem("polarSimApiUrl") || location.origin).replace(/\/$/, "");
const state = {
  snapshot: null,
  deviceFaults: [],
  measurementFaults: [],
  modes: [],
  weatherPoints: [],
  loadPoints: [],
  curveSeries: {},
  isCurveDragging: false,
  settingsLoaded: false,
  activeFaultTab: "devices",
  modeFilter: { dev_type: "all", dev_name: "" },
  runtimeLogs: [],
  lastRuntimeLogKey: "",
};

const $ = (id) => document.getElementById(id);
const MODE_OPTIONS = ["PQ", "PV", "PH", "V"];
const CURVE_POINT_COUNT = 9760;
const CURVE_DURATION_MINUTES = 1440;
const CURVE_META = [
  { key: "wind_speed_mps", label: "风速", color: "#008c8c", min: 0, max: 50, digits: 2, unit: "m/s" },
  { key: "solar_irradiance_w_m2", label: "太阳辐照", color: "#b87500", min: 0, max: 1100, digits: 1, unit: "W/m2" },
  { key: "air_temp_c", label: "气温", color: "#2b6b7f", min: -50, max: 10, digits: 2, unit: "℃" },
  { key: "load_kw", label: "负荷", color: "#c93a3a", min: 0, max: 500, digits: 2, unit: "kW" },
];
const CURVE_PLOT = { left: 58, right: 24, top: 46, bottom: 34 };

function pageFromHash() {
  const fallback = document.querySelector(".app-shell")?.dataset.defaultPage || "overview";
  return (location.hash || "").replace("#", "") || fallback;
}

function showPage(page, updateHash = true) {
  const sections = Array.from(document.querySelectorAll("[data-page]"));
  const target = sections.some((section) => section.dataset.page === page) ? page : "overview";
  sections.forEach((section) => section.classList.toggle("is-active", section.dataset.page === target));
  document.querySelectorAll("[data-nav-page]").forEach((button) => {
    button.classList.toggle("is-active", button.dataset.navPage === target);
  });
  if (updateHash && location.hash !== `#${target}`) {
    history.replaceState(null, "", `#${target}`);
  }
  if (target === "curves" && Object.keys(state.curveSeries).length) {
    requestAnimationFrame(() => {
      resizeCurveCanvas();
      drawCurves();
      renderHourlyTable();
    });
  }
}

function initPageNavigation() {
  document.querySelectorAll("[data-nav-page]").forEach((button) => {
    button.addEventListener("click", () => showPage(button.dataset.navPage));
  });
  window.addEventListener("hashchange", () => showPage(pageFromHash(), false));
  showPage(pageFromHash(), false);
}

async function api(path, options = {}) {
  const response = await fetch(`${apiBase}${path}`, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[char]
  ));
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function pointMinute(index) {
  return (index / Math.max(1, CURVE_POINT_COUNT - 1)) * CURVE_DURATION_MINUTES;
}

function pointIndexFromMinute(minute) {
  return Math.round((minute / CURVE_DURATION_MINUTES) * (CURVE_POINT_COUNT - 1));
}

function curveValueAtMinute(key, minute) {
  const series = state.curveSeries[key] || [];
  return series[clamp(pointIndexFromMinute(minute), 0, series.length - 1)] || 0;
}

function generateCurves(jitter = 0) {
  state.curveSeries = Object.fromEntries(CURVE_META.map((meta) => [meta.key, new Array(CURVE_POINT_COUNT)]));
  const windPeak = 38 + jitter;
  const solarPeak = 720;
  const tempMean = -18;
  const loadBase = 180;
  for (let i = 0; i < CURVE_POINT_COUNT; i += 1) {
    const minute = pointMinute(i);
    const day = minute / CURVE_DURATION_MINUTES;
    const gust = Math.sin(day * Math.PI * 2 * 5 + 0.8) * 4 + Math.sin(day * Math.PI * 2 * 11) * 2;
    const wind = clamp(windPeak * (0.58 + 0.28 * Math.sin(day * Math.PI * 2 - 0.7)) + gust, 0, 50);
    const sunShape = Math.max(0, Math.sin((day - 0.25) * Math.PI * 2));
    const temp = tempMean + 6 * Math.sin((day - 0.33) * Math.PI * 2);
    const load = loadBase * (0.84 + 0.18 * Math.sin((day - 0.18) * Math.PI * 2) + 0.08 * Math.sin(day * Math.PI * 8));
    state.curveSeries.wind_speed_mps[i] = Number(wind.toFixed(2));
    state.curveSeries.solar_irradiance_w_m2[i] = Number((solarPeak * sunShape).toFixed(1));
    state.curveSeries.air_temp_c[i] = Number(temp.toFixed(2));
    state.curveSeries.load_kw[i] = Number(Math.max(20, load).toFixed(2));
  }
  syncCurvePayload();
  drawCurves();
  renderHourlyTable();
}

function syncCurvePayload() {
  state.weatherPoints = [];
  state.loadPoints = [];
  for (let i = 0; i < CURVE_POINT_COUNT; i += 1) {
    const minute = Number(pointMinute(i).toFixed(4));
    const day = minute / CURVE_DURATION_MINUTES;
    state.weatherPoints.push({
      minute,
      wind_speed_mps: roundCurveValue("wind_speed_mps", state.curveSeries.wind_speed_mps[i]),
      air_temp_c: roundCurveValue("air_temp_c", state.curveSeries.air_temp_c[i]),
      air_pressure_hpa: Number((955 + 10 * Math.sin(day * Math.PI * 2 + 0.4)).toFixed(2)),
      solar_irradiance_w_m2: roundCurveValue("solar_irradiance_w_m2", state.curveSeries.solar_irradiance_w_m2[i]),
      humidity_pct: Number((68 + 9 * Math.sin(day * Math.PI * 2 + 2.2)).toFixed(2)),
    });
    state.loadPoints.push({ minute, p_kw: roundCurveValue("load_kw", state.curveSeries.load_kw[i]) });
  }
}

function roundCurveValue(key, value) {
  const meta = CURVE_META.find((item) => item.key === key);
  return Number(clamp(Number(value), meta.min, meta.max).toFixed(meta.digits));
}

function resizeCurveCanvas() {
  const canvas = $("curveEditorChart");
  if (!canvas) return false;
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(320, Math.round(rect.width || canvas.clientWidth || canvas.width));
  const height = Math.max(240, Math.round(rect.height || canvas.clientHeight || canvas.height));
  if (canvas.width === width && canvas.height === height) return false;
  canvas.width = width;
  canvas.height = height;
  return true;
}

function curvePlot(canvas) {
  if (canvas.width < 640) {
    return { left: 34, right: 12, top: 58, bottom: 30 };
  }
  return CURVE_PLOT;
}

function valueToY(value, meta, canvas) {
  const plot = curvePlot(canvas);
  const top = plot.top;
  const bottom = canvas.height - plot.bottom;
  const ratio = (clamp(value, meta.min, meta.max) - meta.min) / (meta.max - meta.min);
  return bottom - ratio * (bottom - top);
}

function yToValue(y, meta, canvas) {
  const plot = curvePlot(canvas);
  const top = plot.top;
  const bottom = canvas.height - plot.bottom;
  const ratio = (bottom - clamp(y, top, bottom)) / (bottom - top);
  return roundCurveValue(meta.key, meta.min + ratio * (meta.max - meta.min));
}

function drawCurves() {
  const canvas = $("curveEditorChart");
  if (!canvas) return;
  resizeCurveCanvas();
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  const plot = curvePlot(canvas);
  const left = plot.left;
  const right = width - plot.right;
  const top = plot.top;
  const bottom = height - plot.bottom;
  const hourStep = width < 480 ? 4 : width < 820 ? 3 : 2;
  const legendColumns = width < 560 ? 2 : CURVE_META.length;
  const legendColumnWidth = (right - left) / legendColumns;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#fcfeff";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "#d8e1e5";
  ctx.lineWidth = 1;
  ctx.font = "12px Microsoft YaHei, Arial";
  ctx.fillStyle = "#63717a";
  for (let i = 0; i <= 5; i += 1) {
    const y = top + i * ((bottom - top) / 5);
    ctx.beginPath();
    ctx.moveTo(left, y);
    ctx.lineTo(right, y);
    ctx.stroke();
  }
  for (let hour = 0; hour <= 24; hour += hourStep) {
    const x = left + (hour / 24) * (right - left);
    ctx.strokeStyle = hour % 6 === 0 ? "#c9d6dc" : "#e7eef1";
    ctx.beginPath();
    ctx.moveTo(x, top);
    ctx.lineTo(x, bottom);
    ctx.stroke();
    ctx.fillStyle = "#63717a";
    ctx.fillText(`${String(hour).padStart(2, "0")}:00`, x - 14, height - 12);
  }
  CURVE_META.forEach((meta, metaIndex) => {
    const values = state.curveSeries[meta.key] || [];
    const stride = Math.max(1, Math.floor(values.length / Math.max(1, (right - left) * 1.4)));
    ctx.strokeStyle = meta.color;
    ctx.lineWidth = meta.key === $("activeCurve").value ? 3 : 2;
    ctx.beginPath();
    for (let i = 0; i < values.length; i += stride) {
      const x = left + (i / Math.max(1, values.length - 1)) * (right - left);
      const y = valueToY(values[i], meta, canvas);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    const lastX = right;
    const lastY = valueToY(values[values.length - 1] || 0, meta, canvas);
    ctx.lineTo(lastX, lastY);
    ctx.stroke();
    const legendX = left + (metaIndex % legendColumns) * legendColumnWidth;
    const legendY = 20 + Math.floor(metaIndex / legendColumns) * 16;
    ctx.fillStyle = meta.color;
    ctx.fillRect(legendX, legendY, 18, 3);
    ctx.fillStyle = "#63717a";
    ctx.fillText(`${meta.label} (${meta.unit})`, legendX + 26, legendY + 4);
  });
}

function pointerPositionOnCanvas(event) {
  const canvas = $("curveEditorChart");
  const rect = canvas.getBoundingClientRect();
  return {
    x: (event.clientX - rect.left) * (canvas.width / rect.width),
    y: (event.clientY - rect.top) * (canvas.height / rect.height),
  };
}

function applyCurveDrag(event) {
  const canvas = $("curveEditorChart");
  const activeKey = $("activeCurve").value;
  const meta = CURVE_META.find((item) => item.key === activeKey);
  const values = state.curveSeries[activeKey] || [];
  if (!canvas || !meta || !values.length) return;
  const pos = pointerPositionOnCanvas(event);
  const plot = curvePlot(canvas);
  const left = plot.left;
  const right = canvas.width - plot.right;
  const index = clamp(Math.round(((pos.x - left) / (right - left)) * (CURVE_POINT_COUNT - 1)), 0, CURVE_POINT_COUNT - 1);
  const targetValue = yToValue(pos.y, meta, canvas);
  const brush = Math.max(12, Math.round(CURVE_POINT_COUNT / 300));
  for (let offset = -brush; offset <= brush; offset += 1) {
    const point = index + offset;
    if (point < 0 || point >= values.length) continue;
    const weight = 1 - Math.abs(offset) / (brush + 1);
    values[point] = roundCurveValue(activeKey, values[point] * (1 - weight) + targetValue * weight);
  }
  syncCurvePayload();
  drawCurves();
  renderHourlyTable();
  $("curveStatus").textContent = "已修改";
}

function renderHourlyTable() {
  const container = $("hourlyCurveTable");
  if (!container) return;
  container.innerHTML = `
    <table class="curve-table">
      <thead>
        <tr>
          <th>时刻</th>
          ${CURVE_META.map((meta) => `<th>${meta.label}<small>${meta.unit}</small></th>`).join("")}
        </tr>
      </thead>
      <tbody>
        ${Array.from({ length: 24 }, (_unused, hour) => `
          <tr>
            <td>${String(hour).padStart(2, "0")}:00</td>
            ${CURVE_META.map((meta) => `
              <td
                contenteditable="true"
                data-hour="${hour}"
                data-key="${meta.key}"
              >${roundCurveValue(meta.key, curveValueAtMinute(meta.key, hour * 60))}</td>
            `).join("")}
          </tr>
        `).join("")}
      </tbody>
    </table>`;
}

function applyHourlyTableEdit(cell) {
  const hour = Number(cell.dataset.hour);
  const key = cell.dataset.key;
  const meta = CURVE_META.find((item) => item.key === key);
  const rawValue = Number(cell.textContent);
  if (!meta || !Number.isFinite(rawValue)) {
    renderHourlyTable();
    return;
  }
  const value = roundCurveValue(key, rawValue);
  const start = hour * 60;
  const end = (hour + 1) * 60;
  const values = state.curveSeries[key] || [];
  for (let i = 0; i < values.length; i += 1) {
    const minute = pointMinute(i);
    if (minute >= start && minute < end) {
      values[i] = value;
    }
  }
  syncCurvePayload();
  drawCurves();
  renderHourlyTable();
  $("curveStatus").textContent = "已修改";
}

function initCurveEditor() {
  const canvas = $("curveEditorChart");
  const table = $("hourlyCurveTable");
  if (!canvas || !table) return;
  canvas.addEventListener("pointerdown", (event) => {
    state.isCurveDragging = true;
    canvas.setPointerCapture(event.pointerId);
    applyCurveDrag(event);
  });
  canvas.addEventListener("pointermove", (event) => {
    if (state.isCurveDragging) applyCurveDrag(event);
  });
  window.addEventListener("pointerup", () => {
    state.isCurveDragging = false;
  });
  $("activeCurve").addEventListener("change", drawCurves);
  window.addEventListener("resize", drawCurves);
  table.addEventListener("blur", (event) => {
    if (event.target.matches("[data-hour][data-key]")) {
      applyHourlyTableEdit(event.target);
    }
  }, true);
  table.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && event.target.matches("[data-hour][data-key]")) {
      event.preventDefault();
      event.target.blur();
    }
  });
}

async function refresh() {
  try {
    const snapshot = await api("/api/snapshot");
    state.snapshot = snapshot;
    renderSnapshot(snapshot);
  } catch (error) {
    $("simState").textContent = "offline";
    $("solverInfo").textContent = "连接失败";
  }
}

function renderSnapshot(snapshot) {
  $("simState").textContent = snapshot.clock.state;
  $("simTime").textContent = snapshot.clock.time;
  $("simSpeed").textContent = `x${snapshot.clock.speed}`;
  $("runtimePath").textContent = snapshot.summary.runtime_dir || "runtime";
  $("metricScada").textContent = snapshot.summary.scada_count;
  $("metricCommands").textContent = snapshot.summary.command_count;
  $("metricAlarms").textContent = snapshot.summary.alarm_count;
  $("metricRefresh").textContent = new Date().toLocaleTimeString();
  $("solverInfo").textContent = snapshot.result.solver_info || "待运行";
  $("overviewSolverInfo").textContent = snapshot.result.solver_info || "待运行";
  $("overviewRefresh").textContent = snapshot.clock.time;
  $("overviewCommandCount").textContent = snapshot.summary.command_count;
  appendRuntimeLog(snapshot);
  renderRuntimeLogs();
  renderMeasurementCompareTable();
  if (!state.settingsLoaded) {
    state.deviceFaults = [...(snapshot.settings?.device_faults || [])];
    state.measurementFaults = [...(snapshot.settings?.measurement_faults || [])];
    state.settingsLoaded = true;
  }
  renderCommands(snapshot.commands.history || []);
  renderDevices(snapshot.devices || []);
  renderFaults();
  state.modes = syncModesFromDevices(snapshot.devices || [], [
    ...(snapshot.settings?.modes || []),
    ...state.modes,
  ]);
  renderModes();
}

function renderCommands(history) {
  const box = $("commandInbox");
  box.innerHTML = history.slice(-8).reverse().map((item) => `
    <div class="log-item">
      <strong>${item.source || "student"} · ${item.time || ""}</strong>
      <span>投退 ${item.accepted?.run_status || 0}，设值 ${item.accepted?.set_values || 0}</span>
    </div>
  `).join("") || '<div class="log-item"><span>暂无命令</span></div>';
}

function appendRuntimeLog(snapshot) {
  const clock = snapshot.clock || {};
  const result = snapshot.result || {};
  const summary = snapshot.summary || {};
  const signature = [
    clock.state,
    clock.time,
    clock.speed,
    result.solver_info,
    result.updated,
    result.missing,
    result.overlay_updates,
    summary.scada_count,
    summary.command_count,
    summary.alarm_count,
  ].join("|");
  if (signature === state.lastRuntimeLogKey) return;
  state.lastRuntimeLogKey = signature;
  state.runtimeLogs.unshift({
    record_time: new Date().toLocaleTimeString(),
    sim_time: clock.time || "--",
    state: clock.state || "--",
    speed: clock.speed ?? "--",
    solver_info: result.solver_info || "待运行",
    updated: result.updated ?? 0,
    missing: result.missing ?? 0,
    overlay_updates: result.overlay_updates ?? 0,
    scada_count: summary.scada_count ?? 0,
    command_count: summary.command_count ?? 0,
    alarm_count: summary.alarm_count ?? 0,
  });
  state.runtimeLogs = state.runtimeLogs.slice(0, 200);
}

function renderRuntimeLogs() {
  const container = $("runtimeLogTable");
  if (!container) return;
  $("runtimeLogSummary").textContent = `最近 ${state.runtimeLogs.length} 条`;
  if (!state.runtimeLogs.length) {
    container.innerHTML = '<div class="empty-state">暂无运行日志</div>';
    return;
  }
  container.innerHTML = `
    <table class="runtime-log-table">
      <thead>
        <tr>
          <th>记录时刻</th>
          <th>仿真时刻</th>
          <th>运行状态</th>
          <th>速度</th>
          <th>求解器</th>
          <th>量测</th>
          <th>命令</th>
          <th>告警</th>
          <th>更新/缺失</th>
        </tr>
      </thead>
      <tbody>
        ${state.runtimeLogs.map((item) => `
          <tr>
            <td>${escapeHtml(item.record_time)}</td>
            <td class="mono-cell">${escapeHtml(item.sim_time)}</td>
            <td><span class="status-dot ${item.state === "running" ? "on" : ""}"></span>${escapeHtml(item.state)}</td>
            <td>x${escapeHtml(item.speed)}</td>
            <td>${escapeHtml(item.solver_info)}</td>
            <td>${escapeHtml(item.scada_count)}</td>
            <td>${escapeHtml(item.command_count)}</td>
            <td>${escapeHtml(item.alarm_count)}</td>
            <td>${escapeHtml(item.updated)} / ${escapeHtml(item.missing)}</td>
          </tr>
        `).join("")}
      </tbody>
    </table>`;
}

function formatMeasurementValue(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "--";
  if (Math.abs(number) >= 1000) return number.toFixed(2);
  if (Math.abs(number) >= 10) return number.toFixed(3);
  return number.toFixed(5);
}

function measurementCompareRows(measurements = state.snapshot?.measurements || {}) {
  const rowsByKey = new Map();
  const addRows = (rows, field) => {
    (rows || []).forEach((row) => {
      const key = measurementKey(row);
      const entry = rowsByKey.get(key) || {};
      entry[field] = row;
      rowsByKey.set(key, entry);
    });
  };
  addRows(measurements.definitions, "definition");
  addRows(measurements.real, "real");
  addRows(measurements.scada, "scada");
  return Array.from(rowsByKey.values()).map((entry) => {
    const base = entry.scada || entry.real || entry.definition || {};
    const realValue = entry.real?.value;
    const scadaValue = entry.scada?.value;
    const realNumber = Number(realValue);
    const scadaNumber = Number(scadaValue);
    const diff = Number.isFinite(realNumber) && Number.isFinite(scadaNumber)
      ? scadaNumber - realNumber
      : null;
    return {
      name: base.name,
      dev_type: base.dev_type,
      dev_name: base.dev_name,
      meas_type: base.meas_type,
      weight: base.weight ?? entry.definition?.weight ?? "--",
      valid: base.valid ?? entry.definition?.valid ?? 0,
      real_value: realValue,
      scada_value: scadaValue,
      diff,
    };
  });
}

function renderMeasurementCompareTable() {
  const container = $("measurementCompareTable");
  if (!container) return;
  const rows = measurementCompareRows();
  const validCount = rows.filter((row) => Number(row.valid) === 1).length;
  $("measurementCompareSummary").textContent = `${rows.length} 点 · 有效 ${validCount} 点`;
  if (!rows.length) {
    container.innerHTML = '<div class="empty-state">暂无实时量测数据</div>';
    return;
  }
  container.innerHTML = `
    <table class="measurement-compare-table">
      <thead>
        <tr>
          <th>量测名称</th>
          <th>设备</th>
          <th>量测类型</th>
          <th>真值</th>
          <th>量测值</th>
          <th>偏差</th>
          <th>权重</th>
          <th>状态</th>
        </tr>
      </thead>
      <tbody>
        ${rows.map((row) => {
          const diffClass = row.diff === null || Math.abs(row.diff) < 1e-6 ? "diff-neutral" : "diff-active";
          return `
            <tr>
              <td>${escapeHtml(row.name || "--")}</td>
              <td>${escapeHtml(row.dev_type || "--")}.${escapeHtml(row.dev_name || "--")}</td>
              <td>${escapeHtml(row.meas_type || "--")}</td>
              <td class="numeric-cell">${formatMeasurementValue(row.real_value)}</td>
              <td class="numeric-cell">${formatMeasurementValue(row.scada_value)}</td>
              <td class="numeric-cell ${diffClass}">${row.diff === null ? "--" : formatMeasurementValue(row.diff)}</td>
              <td class="numeric-cell">${escapeHtml(row.weight)}</td>
              <td><span class="status-dot ${Number(row.valid) === 1 ? "on" : ""}"></span>${Number(row.valid) === 1 ? "有效" : "无效"}</td>
            </tr>
          `;
        }).join("")}
      </tbody>
    </table>`;
}

function renderDevices(devices) {
  $("deviceTable").innerHTML = `
    <table>
      <thead><tr><th>设备</th><th>类型</th><th>投运</th><th>状态</th><th>模式</th><th>设值</th></tr></thead>
      <tbody>
        ${devices.slice(0, 12).map((dev) => `
          <tr>
            <td>${dev.dev_name}</td>
            <td>${dev.dev_type}</td>
            <td><span class="status-dot ${dev.run_stat ? "on" : ""}"></span>${dev.run_stat ? "投入" : "退出"}</td>
            <td>${dev.status ? "闭合/可用" : "断开/故障"}</td>
            <td>${dev.mode || "--"}</td>
            <td>${Object.entries(dev.set_values || {}).map(([k, v]) => `${k}=${v}`).join(" ") || "--"}</td>
          </tr>
        `).join("")}
      </tbody>
    </table>`;
}

function setFaultTab(tabName) {
  state.activeFaultTab = tabName;
  document.querySelectorAll("[data-fault-tab]").forEach((button) => {
    const active = button.dataset.faultTab === tabName;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-selected", active ? "true" : "false");
  });
  document.querySelectorAll("[data-fault-panel]").forEach((panel) => {
    panel.classList.toggle("is-active", panel.dataset.faultPanel === tabName);
  });
}

function deviceKey(dev) {
  return `${dev.dev_type}|${dev.dev_name}`;
}

function measurementKey(meas) {
  return `${meas.name}|${meas.dev_type}|${meas.dev_name}|${meas.meas_type}`;
}

function faultDevices() {
  return state.snapshot?.devices || [];
}

function faultMeasurements() {
  const measurements = state.snapshot?.measurements || {};
  return measurements.scada?.length
    ? measurements.scada
    : measurements.definitions?.length
      ? measurements.definitions
      : measurements.real || [];
}

function findDeviceFault(dev) {
  return state.deviceFaults.find((fault) => fault.dev_type === dev.dev_type && fault.dev_name === dev.dev_name);
}

function findMeasurementFault(meas) {
  return state.measurementFaults.find((fault) => measurementFaultMatches(fault, meas));
}

function measurementFaultMatches(fault, meas) {
    if (fault.dev_type && fault.dev_type !== meas.dev_type) return false;
    if (fault.dev_name && fault.dev_name !== meas.dev_name) return false;
    if (fault.meas_type && String(fault.meas_type).toUpperCase() !== String(meas.meas_type).toUpperCase()) return false;
    const target = fault.target || fault.name || "";
    return !target || target === meas.name || target === meas.dev_name || target === measurementKey(meas);
}

function ensureDeviceFault(dev) {
  let fault = findDeviceFault(dev);
  if (!fault) {
    fault = {
      dev_type: dev.dev_type,
      dev_name: dev.dev_name,
      start_minute: 60,
      clear_minute: 120,
      run_stat: 0,
      status: 0,
    };
    state.deviceFaults.push(fault);
  }
  return fault;
}

function ensureMeasurementFault(meas) {
  let fault = findMeasurementFault(meas);
  if (!fault) {
    fault = {
      name: meas.name,
      target: meas.name,
      dev_type: meas.dev_type,
      dev_name: meas.dev_name,
      meas_type: meas.meas_type,
      fault_type: "dead",
      start_minute: 180,
      clear_minute: 240,
      median: meas.value ?? 0,
      bias: 0,
    };
    state.measurementFaults.push(fault);
  }
  return fault;
}

function renderFaults(force = false) {
  const activeEditor = document.activeElement?.closest?.("#deviceFaultTable, #measurementFaultTable");
  if (!force && activeEditor) return;
  renderDeviceFaultTable();
  renderMeasurementFaultTable();
}

function renderDeviceFaultTable() {
  const container = $("deviceFaultTable");
  const devices = faultDevices();
  if (!container) return;
  $("deviceFaultSummary").textContent = `${state.deviceFaults.length} 个故障`;
  container.innerHTML = `
    <table class="fault-editor-table">
      <thead>
        <tr>
          <th>设备类型</th>
          <th>设备名称</th>
          <th>运行状态</th>
          <th>故障状态</th>
          <th>故障启始时刻</th>
          <th>结束时刻</th>
        </tr>
      </thead>
      <tbody>
        ${devices.map((dev, index) => {
          const fault = findDeviceFault(dev);
          const disabled = fault ? "" : "disabled";
          return `
            <tr>
              <td>${escapeHtml(dev.dev_type)}</td>
              <td>${escapeHtml(dev.dev_name)}</td>
              <td><span class="status-dot ${dev.run_stat ? "on" : ""}"></span>${dev.run_stat ? "投入" : "退出"}</td>
              <td>
                <select data-device-index="${index}" data-device-field="faulted">
                  <option value="normal" ${fault ? "" : "selected"}>正常</option>
                  <option value="fault" ${fault ? "selected" : ""}>故障</option>
                </select>
              </td>
              <td><input data-device-index="${index}" data-device-field="start_minute" type="number" min="0" max="1439" value="${fault?.start_minute ?? 60}" ${disabled} /></td>
              <td><input data-device-index="${index}" data-device-field="clear_minute" type="number" min="0" max="1439" value="${fault?.clear_minute ?? 120}" ${disabled} /></td>
            </tr>`;
        }).join("")}
      </tbody>
    </table>`;
}

function renderMeasurementFaultTable() {
  const container = $("measurementFaultTable");
  const measurements = faultMeasurements();
  if (!container) return;
  $("measurementFaultSummary").textContent = `${state.measurementFaults.length} 个故障`;
  container.innerHTML = `
    <table class="fault-editor-table">
      <thead>
        <tr>
          <th>量测名称</th>
          <th>设备</th>
          <th>量测类型</th>
          <th>当前值</th>
          <th>量测状态</th>
          <th>故障启始时刻</th>
          <th>结束时刻</th>
          <th>中值</th>
          <th>误差</th>
        </tr>
      </thead>
      <tbody>
        ${measurements.map((meas, index) => {
          const fault = findMeasurementFault(meas);
          const faultType = fault?.fault_type || "normal";
          const disabled = fault ? "" : "disabled";
          return `
            <tr>
              <td>${escapeHtml(meas.name)}</td>
              <td>${escapeHtml(meas.dev_type)}.${escapeHtml(meas.dev_name)}</td>
              <td>${escapeHtml(meas.meas_type)}</td>
              <td>${meas.value ?? "--"}</td>
              <td>
                <select data-meas-index="${index}" data-meas-field="fault_type">
                  <option value="normal" ${faultType === "normal" ? "selected" : ""}>正常</option>
                  <option value="dead" ${faultType === "dead" ? "selected" : ""}>死数</option>
                  <option value="zero" ${faultType === "zero" ? "selected" : ""}>0值</option>
                </select>
              </td>
              <td><input data-meas-index="${index}" data-meas-field="start_minute" type="number" min="0" max="1439" value="${fault?.start_minute ?? 180}" ${disabled} /></td>
              <td><input data-meas-index="${index}" data-meas-field="clear_minute" type="number" min="0" max="1439" value="${fault?.clear_minute ?? 240}" ${disabled} /></td>
              <td><input data-meas-index="${index}" data-meas-field="median" type="number" step="0.001" value="${fault?.median ?? meas.value ?? 0}" ${disabled} /></td>
              <td><input data-meas-index="${index}" data-meas-field="bias" type="number" step="0.001" value="${fault?.bias ?? 0}" ${disabled} /></td>
            </tr>`;
        }).join("")}
      </tbody>
    </table>`;
}

function updateDeviceFault(index, field, rawValue, shouldRender = true) {
  const dev = faultDevices()[index];
  if (!dev) return;
  if (field === "faulted" && rawValue === "normal") {
    state.deviceFaults = state.deviceFaults.filter((fault) => deviceKey(fault) !== deviceKey(dev));
    renderFaults(true);
    return;
  }
  const fault = ensureDeviceFault(dev);
  if (field === "start_minute" || field === "clear_minute") {
    fault[field] = Number(rawValue);
  }
  if (shouldRender) renderFaults(true);
}

function updateMeasurementFault(index, field, rawValue, shouldRender = true) {
  const meas = faultMeasurements()[index];
  if (!meas) return;
  if (field === "fault_type" && rawValue === "normal") {
    state.measurementFaults = state.measurementFaults.filter((fault) => !measurementFaultMatches(fault, meas));
    renderFaults(true);
    return;
  }
  const fault = ensureMeasurementFault(meas);
  if (field === "fault_type") {
    fault.fault_type = rawValue;
  } else if (field === "start_minute" || field === "clear_minute" || field === "median" || field === "bias") {
    fault[field] = Number(rawValue);
  }
  if (shouldRender) renderFaults(true);
}

function isModeCapableDevice(dev) {
  if (!dev?.dev_type || !dev?.dev_name) return false;
  if (dev.mode !== undefined && String(dev.mode) !== "") return true;
  const raw = dev.raw || {};
  return ["control_type", "mode", "ctrl_mode"].some((column) => raw[column] !== undefined);
}

function syncModesFromDevices(devices, currentModes = []) {
  const currentByKey = new Map();
  currentModes.forEach((item) => {
    if (item?.dev_type && item?.dev_name) {
      currentByKey.set(deviceKey(item), item);
    }
  });
  return devices.filter(isModeCapableDevice).map((dev) => {
    const existing = currentByKey.get(deviceKey(dev));
    const mode = String(existing?.mode ?? existing?.control_type ?? dev.mode ?? "PQ");
    return {
      dev_type: dev.dev_type,
      dev_name: dev.dev_name,
      mode: mode || "PQ",
    };
  });
}

function modeDeviceMap() {
  return new Map((state.snapshot?.devices || []).map((dev) => [deviceKey(dev), dev]));
}

function modeRows() {
  const devices = modeDeviceMap();
  const filter = state.modeFilter || { dev_type: "all", dev_name: "" };
  return state.modes
    .map((item, index) => ({ item, index, device: devices.get(deviceKey(item)) }))
    .filter(({ item }) => {
      if (filter.dev_type && filter.dev_type !== "all" && item.dev_type !== filter.dev_type) return false;
      if (filter.dev_name && item.dev_name !== filter.dev_name) return false;
      return true;
    });
}

function modeOptionsHtml(value) {
  const current = String(value || "PQ");
  const options = MODE_OPTIONS.includes(current)
    ? MODE_OPTIONS
    : [current, ...MODE_OPTIONS.filter((mode) => mode !== current)];
  return options.map((mode) => `
    <option value="${escapeHtml(mode)}" ${mode === current ? "selected" : ""}>${escapeHtml(mode)}</option>
  `).join("");
}

function renderModeDeviceTree() {
  const container = $("modeDeviceTree");
  if (!container) return;
  const filter = state.modeFilter || { dev_type: "all", dev_name: "" };
  const groups = new Map();
  state.modes.forEach((item) => {
    const list = groups.get(item.dev_type) || [];
    list.push(item);
    groups.set(item.dev_type, list);
  });
  const groupEntries = Array.from(groups.entries()).sort(([left], [right]) => left.localeCompare(right));
  $("modeTreeSummary").textContent = `${groupEntries.length} 类 · ${state.modes.length} 台`;
  container.innerHTML = `
    <button
      type="button"
      class="tree-node tree-root ${filter.dev_type === "all" ? "is-active" : ""}"
      data-mode-tree-type="all"
      data-mode-tree-name=""
    >
      <span>全部设备</span>
      <strong>${state.modes.length}</strong>
    </button>
    ${groupEntries.map(([devType, items]) => `
      <div class="tree-group">
        <button
          type="button"
          class="tree-node tree-type ${filter.dev_type === devType && !filter.dev_name ? "is-active" : ""}"
          data-mode-tree-type="${escapeHtml(devType)}"
          data-mode-tree-name=""
        >
          <span>${escapeHtml(devType)}</span>
          <strong>${items.length}</strong>
        </button>
        <div class="tree-children">
          ${items.map((item) => `
            <button
              type="button"
              class="tree-node tree-child ${filter.dev_type === item.dev_type && filter.dev_name === item.dev_name ? "is-active" : ""}"
              data-mode-tree-type="${escapeHtml(item.dev_type)}"
              data-mode-tree-name="${escapeHtml(item.dev_name)}"
            >
              <span>${escapeHtml(item.dev_name)}</span>
              <small>${escapeHtml(item.mode)}</small>
            </button>
          `).join("")}
        </div>
      </div>
    `).join("")}
  `;
}

function renderModeDeviceTable() {
  const container = $("modeDeviceTable");
  if (!container) return;
  const rows = modeRows();
  $("modeTableSummary").textContent = `${rows.length}/${state.modes.length} 台设备`;
  if (!state.modes.length) {
    container.innerHTML = '<div class="empty-state">暂无可设模式设备</div>';
    return;
  }
  if (!rows.length) {
    container.innerHTML = '<div class="empty-state">当前筛选无设备</div>';
    return;
  }
  container.innerHTML = `
    <table class="mode-editor-table">
      <thead>
        <tr>
          <th>设备类型</th>
          <th>设备名称</th>
          <th>当前状态</th>
          <th>设备状态</th>
          <th>当前模式</th>
          <th>运行模式</th>
        </tr>
      </thead>
      <tbody>
        ${rows.map(({ item, index, device }) => {
          const running = Number(device?.run_stat ?? 1) !== 0;
          const available = Number(device?.status ?? 1) !== 0;
          const currentMode = device?.mode || item.mode || "--";
          return `
            <tr>
              <td>${escapeHtml(item.dev_type)}</td>
              <td class="device-name">${escapeHtml(item.dev_name)}</td>
              <td><span class="status-dot ${running ? "on" : ""}"></span>${running ? "投入" : "退出"}</td>
              <td>${available ? "可用/闭合" : "断开/故障"}</td>
              <td>${escapeHtml(currentMode)}</td>
              <td>
                <select data-mode-device-index="${index}" data-mode-field="mode">
                  ${modeOptionsHtml(item.mode)}
                </select>
              </td>
            </tr>
          `;
        }).join("")}
      </tbody>
    </table>`;
}

function renderModes(force = false) {
  const activeEditor = document.activeElement?.closest?.("#modeDeviceTable");
  if (!force && activeEditor) return;
  renderModeDeviceTree();
  renderModeDeviceTable();
}

function setModeFilter(devType, devName = "") {
  state.modeFilter = { dev_type: devType || "all", dev_name: devName || "" };
  renderModes(true);
}

function updateModeValue(index, field, rawValue) {
  if (field !== "mode" || !state.modes[index]) return;
  state.modes[index].mode = rawValue;
  renderModes(true);
}

async function saveCurves() {
  syncCurvePayload();
  await api("/api/curves", {
    method: "POST",
    body: JSON.stringify({ weather: state.weatherPoints, loads: { load_ac_1: state.loadPoints } }),
  });
  $("curveStatus").textContent = "已保存";
}

async function pushSettings() {
  await api("/api/settings", {
    method: "POST",
    body: JSON.stringify({
      device_faults: state.deviceFaults,
      measurement_faults: state.measurementFaults,
      modes: state.modes,
    }),
  });
  await refresh();
}

document.querySelectorAll("[data-clock]").forEach((button) => {
  button.addEventListener("click", async () => {
    await api("/api/clock", { method: "POST", body: JSON.stringify({ action: button.dataset.clock }) });
    await refresh();
  });
});

$("generateDenseCurves").addEventListener("click", () => {
  generateCurves(0);
  $("curveStatus").textContent = "已生成";
});
$("randomCurves").addEventListener("click", () => {
  generateCurves(Math.random() * 8 - 4);
  $("curveStatus").textContent = "本地扰动";
});
$("saveCurves").addEventListener("click", saveCurves);
$("saveDeviceFaults").addEventListener("click", async () => {
  await pushSettings();
  $("deviceFaultSummary").textContent = `已保存 ${state.deviceFaults.length} 个故障`;
});
$("saveMeasurementFaults").addEventListener("click", async () => {
  await pushSettings();
  $("measurementFaultSummary").textContent = `已保存 ${state.measurementFaults.length} 个故障`;
});
document.querySelectorAll("[data-fault-tab]").forEach((button) => {
  button.addEventListener("click", () => setFaultTab(button.dataset.faultTab));
});
$("pushModes").addEventListener("click", pushSettings);
document.addEventListener("click", (event) => {
  const modeTreeButton = event.target.closest("[data-mode-tree-type]");
  if (modeTreeButton) {
    setModeFilter(modeTreeButton.dataset.modeTreeType, modeTreeButton.dataset.modeTreeName || "");
  }
});
document.addEventListener("change", (event) => {
  if (event.target.dataset.modeField !== undefined) {
    updateModeValue(Number(event.target.dataset.modeDeviceIndex), event.target.dataset.modeField, event.target.value);
  }
  if (event.target.dataset.modeIndex !== undefined) {
    state.modes[Number(event.target.dataset.modeIndex)].mode = event.target.value;
  }
  if (event.target.dataset.deviceField !== undefined) {
    updateDeviceFault(Number(event.target.dataset.deviceIndex), event.target.dataset.deviceField, event.target.value);
  }
  if (event.target.dataset.measField !== undefined) {
    updateMeasurementFault(Number(event.target.dataset.measIndex), event.target.dataset.measField, event.target.value);
  }
});
document.addEventListener("input", (event) => {
  if (event.target.dataset.deviceField !== undefined && event.target.tagName === "INPUT") {
    updateDeviceFault(Number(event.target.dataset.deviceIndex), event.target.dataset.deviceField, event.target.value, false);
  }
  if (event.target.dataset.measField !== undefined && event.target.tagName === "INPUT") {
    updateMeasurementFault(Number(event.target.dataset.measIndex), event.target.dataset.measField, event.target.value, false);
  }
});
initPageNavigation();
generateCurves(0);
initCurveEditor();
setFaultTab(state.activeFaultTab);
renderFaults(true);
setInterval(refresh, 2000);
refresh();
