const apiBase = (window.POLAR_SIM_API_URL || localStorage.getItem("polarSimApiUrl") || location.origin).replace(/\/$/, "");
const state = {
  snapshot: null,
  models: [],
  activeModelId: localStorage.getItem("polarSimulatorModelId") || "",
  deviceFaults: [],
  measurementFaults: [],
  modes: [],
  weatherPoints: [],
  loadPoints: [],
  loadPointsByName: {},
  curveSeries: {},
  curveSeriesByMode: {},
  curveMode: localStorage.getItem("polarSimulatorCurveMode") || "year",
  activeCurveKey: "wind_speed_mps",
  selectedCurveKeys: ["wind_speed_mps"],
  curveEditKey: "",
  isCurveDragging: false,
  settingsLoaded: false,
  activeFaultTab: "devices",
  faultDeviceFilter: { dev_type: "all", dev_name: "" },
  faultMeasurementFilter: { dev_type: "all", dev_name: "", key: "" },
  runtimeDeviceFilter: { dev_type: "all", dev_name: "" },
  runtimeTraceHistory: [],
  runtimeTraceWindowMinutes: 60,
  lastRuntimeTraceKey: "",
  measurementCompareFilter: { dev_type: "all", dev_name: "" },
  selectedMeasurementKey: "",
  measurementTraceHistory: [],
  measurementTraceWindowMinutes: 60,
  lastMeasurementTraceKey: "",
  modeFilter: { dev_type: "all", dev_name: "" },
  runtimeLogs: [],
  lastRuntimeLogKey: "",
};

const $ = (id) => document.getElementById(id);
const MODE_OPTIONS = ["PQ", "PV", "PH", "V"];
const CURVE_MODES = {
  year: { key: "year", label: "年曲线", pointCount: 8760, stepMinutes: 60, durationMinutes: 365 * 24 * 60, tableTitle: "年曲线数据表", tableSummary: "1小时间隔 · 可编辑" },
  day: { key: "day", label: "日曲线", pointCount: 1440, stepMinutes: 1, durationMinutes: 24 * 60, tableTitle: "日曲线数据表", tableSummary: "1分钟间隔 · 可编辑" },
};
const CURVE_META = [
  { key: "wind_speed_mps", label: "风速", color: "#008c8c", min: 0, max: 50, digits: 2, unit: "m/s" },
  { key: "solar_irradiance_w_m2", label: "太阳辐照", color: "#b87500", min: 0, max: 1100, digits: 1, unit: "W/m2" },
  { key: "air_temp_c", label: "气温", color: "#2b6b7f", min: -50, max: 10, digits: 2, unit: "℃" },
  { key: "load_kw", label: "负荷", color: "#c93a3a", min: 0, max: 500, digits: 2, unit: "kW" },
];
const ENV_CURVE_KEYS = ["wind_speed_mps", "solar_irradiance_w_m2", "air_temp_c"];
const LOAD_CURVE_META = { label: "负荷", color: "#c93a3a", min: 0, max: 500, digits: 2, unit: "kW" };
const LOAD_CURVE_COLORS = ["#c93a3a", "#8a4fbf", "#23854a", "#d16300", "#4369b2", "#0a8b8b"];
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
      renderCurveEditor(true);
    });
  }
  if (target === "runtime") {
    requestAnimationFrame(() => drawRuntimeTraceChart());
  }
  if (target === "measurements") {
    requestAnimationFrame(() => drawMeasurementTraceChart());
  }
}

function initPageNavigation() {
  document.querySelectorAll("[data-nav-page]").forEach((button) => {
    button.addEventListener("click", () => showPage(button.dataset.navPage));
  });
  window.addEventListener("hashchange", () => showPage(pageFromHash(), false));
  showPage(pageFromHash(), false);
}

function modelScopedPath(path) {
  if (!state.activeModelId) return path;
  const separator = path.includes("?") ? "&" : "?";
  return `${path}${separator}model_id=${encodeURIComponent(state.activeModelId)}`;
}

async function api(path, options = {}) {
  const { modelScoped = true, ...fetchOptions } = options;
  const targetPath = modelScoped ? modelScopedPath(path) : path;
  const response = await fetch(`${apiBase}${targetPath}`, {
    ...fetchOptions,
    headers: { "Content-Type": "application/json", ...(fetchOptions.headers || {}) },
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

function renderModelSelector() {
  const selector = $("modelSelector");
  if (!selector) return;
  const models = state.models.length ? state.models : [{ id: state.activeModelId || "", name: "默认模型" }];
  selector.innerHTML = models.map((model) => `
    <option value="${escapeHtml(model.id)}">${escapeHtml(model.name || model.id)}</option>
  `).join("");
  selector.value = state.activeModelId || models[0]?.id || "";
  selector.disabled = models.length <= 1;
  const active = models.find((model) => model.id === selector.value) || models[0] || {};
  $("activeModelName").textContent = active.name || active.id || "默认模型";
}

function setActiveModel(modelId, shouldRefresh = true) {
  const nextId = modelId || state.models[0]?.id || "";
  if (state.activeModelId === nextId && shouldRefresh) {
    refresh();
    return;
  }
  state.activeModelId = nextId;
  localStorage.setItem("polarSimulatorModelId", nextId);
  state.snapshot = null;
  state.settingsLoaded = false;
  state.deviceFaults = [];
  state.measurementFaults = [];
  state.modes = [];
  state.runtimeLogs = [];
  state.lastRuntimeLogKey = "";
  state.runtimeTraceHistory = [];
  state.lastRuntimeTraceKey = "";
  state.measurementTraceHistory = [];
  state.lastMeasurementTraceKey = "";
  state.selectedMeasurementKey = "";
  state.modeFilter = { dev_type: "all", dev_name: "" };
  state.faultDeviceFilter = { dev_type: "all", dev_name: "" };
  state.faultMeasurementFilter = { dev_type: "all", dev_name: "", key: "" };
  state.runtimeDeviceFilter = { dev_type: "all", dev_name: "" };
  state.measurementCompareFilter = { dev_type: "all", dev_name: "" };
  state.activeCurveKey = "wind_speed_mps";
  state.selectedCurveKeys = ["wind_speed_mps"];
  state.curveEditKey = "";
  state.curveSeries = {};
  state.curveSeriesByMode = {};
  generateCurves(0, state.curveMode, false);
  renderModelSelector();
  if (shouldRefresh) refresh();
}

async function loadModels() {
  try {
    const catalog = await api("/api/models", { modelScoped: false });
    state.models = Array.isArray(catalog.models) ? catalog.models : [];
    const preferred = state.activeModelId || catalog.active_model_id || state.models[0]?.id || "";
    const exists = state.models.some((model) => model.id === preferred);
    setActiveModel(exists ? preferred : state.models[0]?.id || "", false);
  } catch (_error) {
    state.models = [];
    renderModelSelector();
  }
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[char]
  ));
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function curveModeConfig(mode = state.curveMode) {
  return CURVE_MODES[mode] || CURVE_MODES.year;
}

function curvePointCount(mode = state.curveMode) {
  return curveModeConfig(mode).pointCount;
}

function curveDurationMinutes(mode = state.curveMode) {
  return curveModeConfig(mode).durationMinutes;
}

function curveStepMinutes(mode = state.curveMode) {
  return curveModeConfig(mode).stepMinutes;
}

function pointMinute(index) {
  return index * curveStepMinutes();
}

function pointIndexFromMinute(minute) {
  const config = curveModeConfig();
  const boundedMinute = clamp(Number(minute) || 0, 0, Math.max(0, config.durationMinutes - config.stepMinutes));
  return clamp(Math.round(boundedMinute / config.stepMinutes), 0, config.pointCount - 1);
}

function curveValueAtMinute(key, minute) {
  const series = state.curveSeries[key] || [];
  return series[clamp(pointIndexFromMinute(minute), 0, series.length - 1)] || 0;
}

function loadCurveKey(devName) {
  return `load:${devName || "load_ac_1"}`;
}

function loadNameFromCurveKey(key) {
  return String(key || "").replace(/^load:/, "") || "load_ac_1";
}

function activeCurveKey() {
  return state.activeCurveKey || $("activeCurve")?.value || "wind_speed_mps";
}

function allLoadCurveKeys() {
  return curveLoadDevices().map((dev) => loadCurveKey(dev.dev_name));
}

function allCurveKeys() {
  return [...ENV_CURVE_KEYS, ...allLoadCurveKeys()];
}

function curveLoadDevices() {
  const devices = (state.snapshot?.devices || [])
    .filter((dev) => ["ACLoad", "DCLoad"].includes(dev.dev_type) && dev.dev_name)
    .map((dev) => ({ dev_type: dev.dev_type, dev_name: dev.dev_name }));
  const unique = new Map();
  devices.forEach((dev) => unique.set(`${dev.dev_type}|${dev.dev_name}`, dev));
  const loads = Array.from(unique.values()).sort((left, right) => left.dev_name.localeCompare(right.dev_name));
  return loads.length ? loads : [{ dev_type: "ACLoad", dev_name: "load_ac_1" }];
}

function curveMetaForKey(key) {
  const meta = CURVE_META.find((item) => item.key === key);
  if (meta) return meta;
  if (String(key).startsWith("load:")) {
    const devName = loadNameFromCurveKey(key);
    const loadIndex = Math.max(0, allLoadCurveKeys().indexOf(key));
    const color = LOAD_CURVE_COLORS[loadIndex % LOAD_CURVE_COLORS.length];
    return { ...LOAD_CURVE_META, key, label: devName, color };
  }
  return CURVE_META[0];
}

function activeLoadCurveKey() {
  const key = activeCurveKey();
  if (key.startsWith("load:")) return key;
  return loadCurveKey(curveLoadDevices()[0]?.dev_name);
}

function selectedCurveKeys() {
  const available = new Set(allCurveKeys());
  const selected = Array.from(new Set(state.selectedCurveKeys || []))
    .filter((key) => available.has(key));
  const activeKey = activeCurveKey();
  if (!selected.length && available.has(activeKey)) selected.push(activeKey);
  if (!selected.length) selected.push("wind_speed_mps");
  state.selectedCurveKeys = selected;
  if (!selected.includes(activeKey)) {
    state.activeCurveKey = selected[selected.length - 1];
  }
  return selected;
}

function visibleCurveKeys() {
  return selectedCurveKeys();
}

function visibleCurveMetas() {
  return visibleCurveKeys().map(curveMetaForKey);
}

function resampleSeries(values, nextLength, fallbackValue) {
  if (values?.length === nextLength) return values;
  if (!values?.length) return new Array(nextLength).fill(fallbackValue);
  if (nextLength <= 1) return [values[0] ?? fallbackValue];
  const lastSource = Math.max(1, values.length - 1);
  return Array.from({ length: nextLength }, (_unused, index) => {
    const sourceIndex = Math.round((index / Math.max(1, nextLength - 1)) * lastSource);
    return values[sourceIndex] ?? fallbackValue;
  });
}

function normalizeCurveSeriesLength(key, fallbackValue) {
  const nextLength = curvePointCount();
  const changed = state.curveSeries[key]?.length !== nextLength;
  state.curveSeries[key] = resampleSeries(state.curveSeries[key], nextLength, fallbackValue);
  return changed;
}

function loadCurveSeriesTemplate() {
  const firstLoadKey = loadCurveKey(curveLoadDevices()[0]?.dev_name);
  return resampleSeries(state.curveSeries.load_kw || state.curveSeries[firstLoadKey], curvePointCount(), 120);
}

function ensureCurveLoadSeries() {
  const template = loadCurveSeriesTemplate();
  let changed = false;
  curveLoadDevices().forEach((dev) => {
    const key = loadCurveKey(dev.dev_name);
    if (!state.curveSeries[key]) {
      state.curveSeries[key] = [...template];
      changed = true;
    } else if (state.curveSeries[key].length !== curvePointCount()) {
      state.curveSeries[key] = resampleSeries(state.curveSeries[key], curvePointCount(), 120);
      changed = true;
    }
  });
  const activeKey = activeCurveKey();
  if (activeKey.startsWith("load:") && !state.curveSeries[activeKey]) {
    setActiveCurve(loadCurveKey(curveLoadDevices()[0]?.dev_name), false);
    changed = true;
  }
  return changed;
}

function ensureCurveSeries() {
  let changed = false;
  ENV_CURVE_KEYS.forEach((key) => {
    changed = normalizeCurveSeriesLength(key, curveMetaForKey(key).min) || changed;
  });
  changed = ensureCurveLoadSeries() || changed;
  return changed;
}

function saveCurrentCurveModeSeries() {
  if (!state.curveMode || !Object.keys(state.curveSeries || {}).length) return;
  state.curveSeriesByMode[state.curveMode] = state.curveSeries;
}

function setCurveMode(mode, shouldRender = true) {
  const nextMode = CURVE_MODES[mode] ? mode : "year";
  saveCurrentCurveModeSeries();
  state.curveMode = nextMode;
  localStorage.setItem("polarSimulatorCurveMode", nextMode);
  state.curveEditKey = "";
  if (state.curveSeriesByMode[nextMode]) {
    state.curveSeries = state.curveSeriesByMode[nextMode];
    ensureCurveSeries();
    syncCurvePayload(false);
  } else {
    generateCurves(0, nextMode, false);
  }
  if (shouldRender) {
    renderCurveEditor(true);
  }
}

function renderCurveModeControls() {
  document.querySelectorAll("[data-curve-mode]").forEach((button) => {
    const active = button.dataset.curveMode === state.curveMode;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-pressed", active ? "true" : "false");
  });
}

function updateCurveModeLabels() {
  const config = curveModeConfig();
  const pointCount = $("curvePointCount");
  const tableTitle = $("curveTableTitle");
  const tableSummary = $("curveTableSummary");
  if (pointCount) pointCount.textContent = `${config.pointCount}点`;
  if (tableTitle) tableTitle.textContent = config.tableTitle;
  if (tableSummary) tableSummary.textContent = config.tableSummary;
}

function curveFamilyKeys(family) {
  if (family === "environment") return [...ENV_CURVE_KEYS];
  if (family === "load") return allLoadCurveKeys();
  return [];
}

function selectedCurveLabel() {
  const selected = selectedCurveKeys();
  const editKey = curveEditKey(selected);
  const selectedLabel = selected.length <= 1 ? curveMetaForKey(selected[0]).label : `已选${selected.length}条`;
  return editKey && selected.length > 1 ? `${selectedLabel} · ${curveMetaForKey(editKey).label}` : selectedLabel;
}

function setSelectedCurves(keys, activeKey = keys?.[keys.length - 1], shouldRender = true) {
  const available = new Set(allCurveKeys());
  const selected = Array.from(new Set(keys || [])).filter((key) => available.has(key));
  if (!selected.length) selected.push("wind_speed_mps");
  const nextActiveKey = selected.includes(activeKey) ? activeKey : selected[selected.length - 1];
  state.selectedCurveKeys = selected;
  state.activeCurveKey = nextActiveKey || "wind_speed_mps";
  if (state.curveEditKey && !selected.includes(state.curveEditKey)) {
    state.curveEditKey = "";
  }
  const activeInput = $("activeCurve");
  if (activeInput) activeInput.value = state.activeCurveKey;
  if (shouldRender) {
    renderCurveTree();
    drawCurves();
    renderHourlyTable();
  }
}

function toggleCurveSelection(key, shouldRender = true) {
  const selected = selectedCurveKeys();
  const next = selected.includes(key)
    ? selected.filter((item) => item !== key)
    : [...selected, key];
  setSelectedCurves(next.length ? next : selected, key, shouldRender);
}

function selectCurveFamily(family, shouldRender = true) {
  const familyKeys = curveFamilyKeys(family);
  setSelectedCurves(familyKeys, familyKeys[0], shouldRender);
}

function curveEditKey(selectedKeys = selectedCurveKeys()) {
  const editKey = state.curveEditKey || "";
  if (editKey && selectedKeys.includes(editKey) && (state.curveSeries[editKey] || []).length) {
    return editKey;
  }
  if (editKey) state.curveEditKey = "";
  return "";
}

function setCurveEditKey(key, shouldRender = true) {
  const selected = selectedCurveKeys();
  const nextKey = selected.includes(key) ? key : "";
  state.curveEditKey = nextKey;
  if (nextKey) {
    state.activeCurveKey = nextKey;
    const activeInput = $("activeCurve");
    if (activeInput) activeInput.value = nextKey;
  }
  if (shouldRender) {
    renderCurveTree();
    drawCurves();
  }
}

function cancelCurveEditSelection() {
  state.curveEditKey = "";
  state.isCurveDragging = false;
  renderCurveTree();
  drawCurves();
}

function renderCurveTree() {
  const container = $("curveTree");
  if (!container) return;
  const activeKey = activeCurveKey();
  const selectedKeys = selectedCurveKeys();
  const editKey = curveEditKey(selectedKeys);
  const selectedSet = new Set(selectedKeys);
  const loadDevices = curveLoadDevices();
  const loadKeys = allLoadCurveKeys();
  const envSelected = ENV_CURVE_KEYS.every((key) => selectedSet.has(key))
    && selectedKeys.every((key) => ENV_CURVE_KEYS.includes(key));
  const loadSelected = loadKeys.every((key) => selectedSet.has(key))
    && selectedKeys.every((key) => loadKeys.includes(key));
  const envPartial = ENV_CURVE_KEYS.some((key) => selectedSet.has(key));
  const loadPartial = loadKeys.some((key) => selectedSet.has(key));
  $("curveTreeSummary").textContent = `${ENV_CURVE_KEYS.length + loadDevices.length} 条`;
  $("activeCurve").value = activeKey;
  $("activeCurveLabel").textContent = selectedCurveLabel();
  container.innerHTML = `
    <div class="tree-group">
      <button
        type="button"
        class="tree-node tree-type ${envSelected ? "is-active" : envPartial ? "is-parent-active" : ""}"
        data-curve-tree-type="environment"
        data-curve-family="environment"
        aria-pressed="${envSelected ? "true" : "false"}"
      >
        <span>环境曲线</span>
        <strong>${ENV_CURVE_KEYS.length}</strong>
      </button>
      <div class="tree-children">
        ${ENV_CURVE_KEYS.map((key) => {
          const meta = curveMetaForKey(key);
          const shortLabel = key === "wind_speed_mps" ? "风" : key === "solar_irradiance_w_m2" ? "光" : "温";
          return `
            <button
              type="button"
              class="tree-node tree-child ${selectedSet.has(key) ? "is-active" : ""} ${editKey === key ? "is-edit-target" : ""}"
              data-curve-tree-type="environment"
              data-curve-key="${escapeHtml(key)}"
              aria-pressed="${selectedSet.has(key) ? "true" : "false"}"
            >
              <span>${shortLabel}</span>
              <small>${escapeHtml(meta.unit)}</small>
            </button>
          `;
        }).join("")}
      </div>
    </div>
    <div class="tree-group">
      <button
        type="button"
        class="tree-node tree-type ${loadSelected ? "is-active" : loadPartial ? "is-parent-active" : ""}"
        data-curve-tree-type="load"
        data-curve-family="load"
        aria-pressed="${loadSelected ? "true" : "false"}"
      >
        <span>负荷曲线</span>
        <strong>${loadDevices.length}</strong>
      </button>
      <div id="curveLoadTree" class="tree-children">
        ${loadDevices.map((dev) => {
          const key = loadCurveKey(dev.dev_name);
          return `
            <button
              type="button"
              class="tree-node tree-child ${selectedSet.has(key) ? "is-active" : ""} ${editKey === key ? "is-edit-target" : ""}"
              data-curve-tree-type="load"
              data-curve-key="${escapeHtml(key)}"
              aria-pressed="${selectedSet.has(key) ? "true" : "false"}"
            >
              <span>${escapeHtml(dev.dev_name)}</span>
              <small>${escapeHtml(dev.dev_type)}</small>
            </button>
          `;
        }).join("")}
      </div>
    </div>
  `;
}

function setActiveCurve(key, shouldRender = true) {
  const nextKey = key || "wind_speed_mps";
  setSelectedCurves([nextKey], nextKey, shouldRender);
}

function renderCurveEditor(force = false) {
  const seriesChanged = ensureCurveSeries();
  if (seriesChanged) syncCurvePayload(false);
  renderCurveTree();
  renderCurveModeControls();
  updateCurveModeLabels();
  const activeEditor = document.activeElement?.closest?.("#hourlyCurveTable");
  if (!force && activeEditor) return;
  drawCurves();
  renderHourlyTable();
}

function generateCurves(jitter = 0, mode = state.curveMode, shouldRender = true) {
  state.curveMode = CURVE_MODES[mode] ? mode : "year";
  const config = curveModeConfig();
  const pointCount = curvePointCount();
  state.curveSeries = Object.fromEntries(ENV_CURVE_KEYS.map((key) => [key, new Array(pointCount)]));
  const windPeak = 38 + jitter;
  const solarPeak = 720;
  const tempMean = -18;
  const loadBase = 180;
  const loadDevices = curveLoadDevices();
  loadDevices.forEach((dev) => {
    state.curveSeries[loadCurveKey(dev.dev_name)] = new Array(pointCount);
  });
  for (let i = 0; i < pointCount; i += 1) {
    const minute = pointMinute(i);
    const day = (minute % (24 * 60)) / (24 * 60);
    const year = minute / config.durationMinutes;
    const season = state.curveMode === "year" ? Math.sin((year - 0.18) * Math.PI * 2) : 0;
    const gust = Math.sin(day * Math.PI * 2 * 5 + 0.8) * 4 + Math.sin(day * Math.PI * 2 * 11 + year * 9) * 2;
    const wind = clamp(windPeak * (0.58 + 0.28 * Math.sin(day * Math.PI * 2 - 0.7) + 0.10 * season) + gust, 0, 50);
    const daylight = Math.max(0, Math.sin((day - 0.25) * Math.PI * 2));
    const solarSeason = state.curveMode === "year" ? clamp(0.58 + 0.42 * season, 0.05, 1.0) : 1.0;
    const tempSeason = state.curveMode === "year" ? 9 * season : 0;
    const sunShape = daylight * solarSeason;
    const temp = tempMean + tempSeason + 6 * Math.sin((day - 0.33) * Math.PI * 2);
    const load = loadBase * (0.84 + 0.18 * Math.sin((day - 0.18) * Math.PI * 2) + 0.08 * Math.sin(day * Math.PI * 8));
    state.curveSeries.wind_speed_mps[i] = Number(wind.toFixed(2));
    state.curveSeries.solar_irradiance_w_m2[i] = Number((solarPeak * sunShape).toFixed(1));
    state.curveSeries.air_temp_c[i] = Number(temp.toFixed(2));
    loadDevices.forEach((dev, loadIndex) => {
      const offset = 1 + loadIndex * 0.035;
      state.curveSeries[loadCurveKey(dev.dev_name)][i] = Number(Math.max(20, load * offset).toFixed(2));
    });
  }
  state.curveSeries.load_kw = [...state.curveSeries[loadCurveKey(loadDevices[0]?.dev_name)]];
  state.curveSeriesByMode[state.curveMode] = state.curveSeries;
  syncCurvePayload(false);
  if (shouldRender) renderCurveEditor(true);
}

function syncCurvePayload(shouldStoreSeries = true) {
  ensureCurveSeries();
  const config = curveModeConfig();
  state.weatherPoints = [];
  state.loadPoints = [];
  state.loadPointsByName = {};
  curveLoadDevices().forEach((dev) => {
    state.loadPointsByName[dev.dev_name] = [];
  });
  for (let i = 0; i < config.pointCount; i += 1) {
    const minute = Number(pointMinute(i).toFixed(4));
    const year = minute / config.durationMinutes;
    state.weatherPoints.push({
      minute,
      wind_speed_mps: roundCurveValue("wind_speed_mps", state.curveSeries.wind_speed_mps[i]),
      air_temp_c: roundCurveValue("air_temp_c", state.curveSeries.air_temp_c[i]),
      air_pressure_hpa: Number((955 + 10 * Math.sin(year * Math.PI * 2 + 0.4)).toFixed(2)),
      solar_irradiance_w_m2: roundCurveValue("solar_irradiance_w_m2", state.curveSeries.solar_irradiance_w_m2[i]),
      humidity_pct: Number((68 + 9 * Math.sin(year * Math.PI * 2 + 2.2)).toFixed(2)),
    });
    curveLoadDevices().forEach((dev, loadIndex) => {
      const key = loadCurveKey(dev.dev_name);
      const point = { minute, p_kw: roundCurveValue(key, state.curveSeries[key]?.[i] ?? 0) };
      state.loadPointsByName[dev.dev_name].push(point);
      if (loadIndex === 0) state.loadPoints.push(point);
    });
  }
  if (shouldStoreSeries) state.curveSeriesByMode[state.curveMode] = state.curveSeries;
}

function roundCurveValue(key, value) {
  const meta = curveMetaForKey(key);
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

function drawCurveXAxis(ctx, canvas, plot) {
  const width = canvas.width;
  const height = canvas.height;
  const left = plot.left;
  const right = width - plot.right;
  const top = plot.top;
  const bottom = height - plot.bottom;
  if (state.curveMode === "year") {
    const monthStarts = [
      ["01月", 0],
      ["02月", 31],
      ["03月", 59],
      ["04月", 90],
      ["05月", 120],
      ["06月", 151],
      ["07月", 181],
      ["08月", 212],
      ["09月", 243],
      ["10月", 273],
      ["11月", 304],
      ["12月", 334],
    ];
    const monthStep = width < 560 ? 3 : width < 900 ? 2 : 1;
    monthStarts.forEach(([label, day], index) => {
      if (index % monthStep !== 0) return;
      const x = left + (day / 365) * (right - left);
      ctx.strokeStyle = index % 3 === 0 ? "#c9d6dc" : "#e7eef1";
      ctx.beginPath();
      ctx.moveTo(x, top);
      ctx.lineTo(x, bottom);
      ctx.stroke();
      ctx.fillStyle = "#63717a";
      ctx.fillText(label, x - 12, height - 12);
    });
    ctx.strokeStyle = "#c9d6dc";
    ctx.beginPath();
    ctx.moveTo(right, top);
    ctx.lineTo(right, bottom);
    ctx.stroke();
    ctx.fillStyle = "#63717a";
    ctx.textAlign = "right";
    ctx.fillText("年末", right, height - 12);
    ctx.textAlign = "left";
    return;
  }
  const hourStep = width < 480 ? 4 : width < 820 ? 3 : 2;
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
  const metas = visibleCurveMetas();
  const editKey = curveEditKey(metas.map((meta) => meta.key));
  const legendColumns = width < 560 ? 2 : metas.length;
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
  drawCurveXAxis(ctx, canvas, plot);
  metas.forEach((meta, metaIndex) => {
    const values = state.curveSeries[meta.key] || [];
    const stride = Math.max(1, Math.floor(values.length / Math.max(1, (right - left) * 1.4)));
    ctx.strokeStyle = meta.color;
    ctx.lineWidth = editKey && meta.key === editKey ? 3.5 : 2;
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

function curvePointIndexFromX(x, canvas) {
  const plot = curvePlot(canvas);
  const left = plot.left;
  const right = canvas.width - plot.right;
  const pointCount = curvePointCount();
  return clamp(Math.round(((x - left) / (right - left)) * (pointCount - 1)), 0, pointCount - 1);
}

function curveKeyAtPointer(event) {
  const canvas = $("curveEditorChart");
  if (!canvas) return "";
  const pos = pointerPositionOnCanvas(event);
  const plot = curvePlot(canvas);
  const left = plot.left;
  const right = canvas.width - plot.right;
  const top = plot.top;
  const bottom = canvas.height - plot.bottom;
  if (pos.x < left || pos.x > right || pos.y < top || pos.y > bottom) return "";
  const index = curvePointIndexFromX(pos.x, canvas);
  const tolerance = canvas.width < 640 ? 18 : 14;
  let bestKey = "";
  let bestDistance = Infinity;
  visibleCurveMetas().forEach((meta) => {
    const values = state.curveSeries[meta.key] || [];
    if (!values.length) return;
    const distance = Math.abs(valueToY(values[index], meta, canvas) - pos.y);
    if (distance < bestDistance) {
      bestDistance = distance;
      bestKey = meta.key;
    }
  });
  return bestDistance <= tolerance ? bestKey : "";
}

function applyCurveDrag(event) {
  const canvas = $("curveEditorChart");
  const editKey = curveEditKey();
  const meta = curveMetaForKey(editKey);
  const values = state.curveSeries[editKey] || [];
  if (!canvas || !meta || !values.length) return;
  const pos = pointerPositionOnCanvas(event);
  const index = curvePointIndexFromX(pos.x, canvas);
  const targetValue = yToValue(pos.y, meta, canvas);
  const brush = Math.max(12, Math.round(curvePointCount() / 300));
  for (let offset = -brush; offset <= brush; offset += 1) {
    const point = index + offset;
    if (point < 0 || point >= values.length) continue;
    const weight = 1 - Math.abs(offset) / (brush + 1);
    values[point] = roundCurveValue(editKey, values[point] * (1 - weight) + targetValue * weight);
  }
  syncCurvePayload();
  drawCurves();
  $("curveStatus").textContent = "已修改";
}

function formatCurveTableTime(minute) {
  if (state.curveMode === "year") {
    const dayOfYear = Math.floor(minute / (24 * 60));
    const hour = Math.floor((minute % (24 * 60)) / 60);
    const monthDays = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let month = 0;
    let day = dayOfYear;
    while (month < monthDays.length - 1 && day >= monthDays[month]) {
      day -= monthDays[month];
      month += 1;
    }
    return `${String(month + 1).padStart(2, "0")}-${String(day + 1).padStart(2, "0")} ${String(hour).padStart(2, "0")}:00`;
  }
  const total = Math.round(minute);
  const hour = Math.floor(total / 60);
  const minutePart = total % 60;
  return `${String(hour).padStart(2, "0")}:${String(minutePart).padStart(2, "0")}`;
}

function renderHourlyTable() {
  const container = $("hourlyCurveTable");
  if (!container) return;
  const metas = visibleCurveMetas();
  const pointCount = curvePointCount();
  container.innerHTML = `
    <table class="curve-table">
      <thead>
        <tr>
          <th>时刻</th>
          ${metas.map((meta) => `<th>${escapeHtml(meta.label)}<small>${escapeHtml(meta.unit)}</small></th>`).join("")}
        </tr>
      </thead>
      <tbody>
        ${Array.from({ length: pointCount }, (_unused, index) => `
          <tr>
            <td>${formatCurveTableTime(pointMinute(index))}</td>
            ${metas.map((meta) => `
              <td
                contenteditable="true"
                data-index="${index}"
                data-key="${escapeHtml(meta.key)}"
              >${roundCurveValue(meta.key, state.curveSeries[meta.key]?.[index] ?? 0)}</td>
            `).join("")}
          </tr>
        `).join("")}
      </tbody>
    </table>`;
}

function applyHourlyTableEdit(cell) {
  const index = Number(cell.dataset.index);
  const key = cell.dataset.key;
  const meta = curveMetaForKey(key);
  const rawValue = Number(cell.textContent);
  if (!meta || !Number.isFinite(rawValue) || !Number.isInteger(index)) {
    renderHourlyTable();
    return;
  }
  const value = roundCurveValue(key, rawValue);
  const values = state.curveSeries[key] || [];
  if (index >= 0 && index < values.length) values[index] = value;
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
    if (event.button === 2) {
      event.preventDefault();
      cancelCurveEditSelection();
      return;
    }
    if (event.button !== 0) return;
    const hitKey = curveKeyAtPointer(event);
    if (!hitKey) return;
    event.preventDefault();
    setCurveEditKey(hitKey);
    state.isCurveDragging = true;
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener("pointermove", (event) => {
    if (state.isCurveDragging) {
      event.preventDefault();
      applyCurveDrag(event);
    }
  });
  canvas.addEventListener("contextmenu", (event) => {
    event.preventDefault();
    cancelCurveEditSelection();
  });
  canvas.addEventListener("pointercancel", cancelCurveEditSelection);
  window.addEventListener("pointerup", () => {
    const wasDragging = state.isCurveDragging;
    state.isCurveDragging = false;
    if (wasDragging) renderHourlyTable();
  });
  $("activeCurve").addEventListener("change", (event) => setActiveCurve(event.target.value));
  window.addEventListener("resize", drawCurves);
  table.addEventListener("blur", (event) => {
    if (event.target.matches("[data-index][data-key]")) {
      applyHourlyTableEdit(event.target);
    }
  }, true);
  table.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && event.target.matches("[data-index][data-key]")) {
      event.preventDefault();
      event.target.blur();
    }
  });
}

function initRuntimeMonitor() {
  const windowSelect = $("runtimeTraceWindow");
  if (windowSelect) {
    state.runtimeTraceWindowMinutes = Number(windowSelect.value) || state.runtimeTraceWindowMinutes;
    windowSelect.addEventListener("change", (event) => {
      state.runtimeTraceWindowMinutes = Number(event.target.value) || 60;
      drawRuntimeTraceChart();
    });
  }
  window.addEventListener("resize", drawRuntimeTraceChart);
}

function initMeasurementMonitor() {
  const windowSelect = $("measurementTraceWindow");
  if (windowSelect) {
    state.measurementTraceWindowMinutes = Number(windowSelect.value) || state.measurementTraceWindowMinutes;
    windowSelect.addEventListener("change", (event) => {
      state.measurementTraceWindowMinutes = Number(event.target.value) || 60;
      drawMeasurementTraceChart();
    });
  }
  window.addEventListener("resize", drawMeasurementTraceChart);
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
  if (snapshot.model?.id && snapshot.model.id !== state.activeModelId) {
    state.activeModelId = snapshot.model.id;
  }
  renderModelSelector();
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
  appendRuntimeTrace(snapshot);
  appendMeasurementTrace(snapshot);
  renderRuntimeLogs();
  renderMeasurementCompareTable();
  if (!state.settingsLoaded) {
    state.deviceFaults = [...(snapshot.settings?.device_faults || [])];
    state.measurementFaults = [...(snapshot.settings?.measurement_faults || [])];
    state.settingsLoaded = true;
  }
  renderCommands(snapshot.commands.history || []);
  renderRuntimeMonitor();
  renderCurveEditor();
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

function numberOrNull(value) {
  if (value === null || value === undefined || value === "") return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function runtimeDevices() {
  return state.snapshot?.devices || [];
}

function runtimeFilterMatches(dev, filter = state.runtimeDeviceFilter || { dev_type: "all", dev_name: "" }) {
  if (filter.dev_type && filter.dev_type !== "all" && dev.dev_type !== filter.dev_type) return false;
  if (filter.dev_name && dev.dev_name !== filter.dev_name) return false;
  return true;
}

function filteredRuntimeDevices(devices = runtimeDevices()) {
  return devices.filter((dev) => runtimeFilterMatches(dev));
}

function runtimeControlMeta(dev) {
  const setValues = dev?.set_values || {};
  const raw = dev?.raw || {};
  const mode = String(dev?.mode || raw.control_type || raw.ctrl_mode || "").toUpperCase();
  const preferred = [];
  if (mode.includes("V")) preferred.push("v_set", "v_ac_set", "v_dc_set");
  if (mode.includes("Q")) preferred.push("q_set", "q_ac_set");
  if (mode.includes("P") || mode.includes("H")) preferred.push("p_ac_set", "p_dc_set", "p_set", "pv0");
  preferred.push(
    "p_ac_set",
    "p_dc_set",
    "p_set",
    "pv0",
    "q_ac_set",
    "q_set",
    "qv0",
    "v_ac_set",
    "v_dc_set",
    "v_set",
    "i_set",
  );
  const candidates = Array.from(new Set(preferred));
  for (const key of candidates) {
    const value = numberOrNull(setValues[key] ?? raw[key]);
    if (value !== null) return runtimeMetaFromSetKey(key, value);
  }
  const soc = numberOrNull(dev?.soc_curr ?? raw.soc_curr ?? raw.soc);
  if (soc !== null) {
    return { key: "soc_curr", label: "soc_curr", kind: "SOC", unit: "%", value: soc };
  }
  return { key: "run_stat", label: "run_stat", kind: "STAT", unit: "", value: numberOrNull(dev?.run_stat) ?? 0 };
}

function runtimeMetaFromSetKey(key, value) {
  const lowerKey = String(key).toLowerCase();
  if (lowerKey.includes("soc")) return { key, label: key, kind: "SOC", unit: "%", value };
  if (lowerKey.startsWith("q") || lowerKey.includes("_q")) return { key, label: key, kind: "Q", unit: "kvar", value };
  if (lowerKey.startsWith("v") || lowerKey.includes("_v")) return { key, label: key, kind: "V", unit: "V", value };
  if (lowerKey.startsWith("i") || lowerKey.includes("_i")) return { key, label: key, kind: "I", unit: "A", value };
  if (lowerKey === "run_stat") return { key, label: key, kind: "STAT", unit: "", value };
  return { key, label: key, kind: "P", unit: "kW", value };
}

function runtimeMeasurementHints(meta) {
  const key = String(meta.key || "").toLowerCase();
  if (key.includes("p_ac")) return ["P_AC", "P_GEN", "P_LOAD", "P_FROM", "P_TO", "P_DC", "P"];
  if (key.includes("p_dc")) return ["P_DC", "P_FROM", "P_TO", "P_GEN", "P_LOAD", "P"];
  if (key.includes("q_ac")) return ["Q_AC", "Q_GEN", "Q_LOAD", "Q_FROM", "Q_TO", "Q"];
  if (key.includes("v_ac")) return ["V_AC", "V_GEN", "V_LOAD", "V_FROM", "V_TO", "V"];
  if (key.includes("v_dc")) return ["V_DC", "V_GEN", "V_FROM", "V_TO", "V"];
  const hints = {
    P: ["P_GEN", "P_LOAD", "P_AC", "P_DC", "P_FROM", "P_TO", "P"],
    Q: ["Q_GEN", "Q_LOAD", "Q_AC", "Q_FROM", "Q_TO", "Q"],
    V: ["V_GEN", "V_LOAD", "V_AC", "V_DC", "V_FROM", "V_TO", "V"],
    I: ["I_GEN", "I_LOAD", "I_AC", "I_DC", "I_FROM", "I_TO", "I"],
    SOC: ["SOC"],
  };
  return hints[meta.kind] || [];
}

function runtimeMeasurementBaseScore(row, dev) {
  if (!row || !dev) return 0;
  const rowType = String(row.dev_type || "");
  const rowName = String(row.dev_name || "");
  const devType = String(dev.dev_type || "");
  const devName = String(dev.dev_name || "");
  const rowLabel = `${row.name || ""} ${rowName}`;
  if (rowType === devType && rowName === devName) return 100;
  if (rowName === devName) return 84;
  if (rowName.startsWith(`${devName}_`) || rowName.startsWith(devName)) return 64;
  if (rowLabel.includes(devName)) return 48;
  return 0;
}

function runtimeMeasurementScore(row, dev, hints) {
  const base = runtimeMeasurementBaseScore(row, dev);
  if (!base) return 0;
  const measType = String(row.meas_type || "").toUpperCase();
  const hintIndex = hints.indexOf(measType);
  if (hintIndex >= 0) return base + 40 - hintIndex;
  if (hints.includes("SOC")) return 0;
  const prefix = measType.split("_")[0];
  const prefixIndex = hints.findIndex((hint) => hint.split("_")[0] === prefix);
  return base + (prefixIndex >= 0 ? 12 - prefixIndex : 0);
}

function runtimeMeasurementPair(dev, meta, measurements = state.snapshot?.measurements || {}) {
  const hints = runtimeMeasurementHints(meta);
  const rows = measurementCompareRows(measurements)
    .map((row) => ({ row, score: runtimeMeasurementScore(row, dev, hints) }))
    .filter((item) => item.score > 0)
    .sort((left, right) => right.score - left.score);
  const best = rows[0]?.row || {};
  return {
    name: best.name || "",
    meas_type: best.meas_type || "",
    real: numberOrNull(best.real_value),
    scada: numberOrNull(best.scada_value),
  };
}

function runtimeDeviceTraceSignal(dev, measurements = state.snapshot?.measurements || {}) {
  const control = runtimeControlMeta(dev);
  const pair = runtimeMeasurementPair(dev, control, measurements);
  return {
    control: control.value,
    real: pair.real,
    scada: pair.scada,
    set_type: control.key,
    signal_kind: control.kind,
    unit: control.unit,
    meas_name: pair.name,
    meas_type: pair.meas_type,
  };
}

function appendRuntimeTrace(snapshot) {
  const clock = snapshot.clock || {};
  const result = snapshot.result || {};
  const summary = snapshot.summary || {};
  const signature = [
    snapshot.model?.id || state.activeModelId,
    clock.absolute_minute ?? clock.minute ?? "",
    clock.time || "",
    result.updated ?? "",
    result.solver_info || "",
    summary.scada_count ?? 0,
  ].join("|");
  if (signature === state.lastRuntimeTraceKey) return;
  state.lastRuntimeTraceKey = signature;
  const point = {
    minute: Number(clock.absolute_minute ?? clock.minute ?? state.runtimeTraceHistory.length) || 0,
    sim_time: clock.time || "--",
    record_time: Date.now(),
    devices: {},
  };
  (snapshot.devices || []).forEach((dev) => {
    point.devices[deviceKey(dev)] = runtimeDeviceTraceSignal(dev, snapshot.measurements || {});
  });
  state.runtimeTraceHistory.push(point);
  state.runtimeTraceHistory = state.runtimeTraceHistory.slice(-3000);
}

function renderRuntimeDeviceTree() {
  const container = $("runtimeDeviceTree");
  if (!container) return;
  const devices = runtimeDevices();
  const filter = state.runtimeDeviceFilter || { dev_type: "all", dev_name: "" };
  const groupEntries = groupedByDeviceType(devices);
  $("runtimeTreeSummary").textContent = `${groupEntries.length} 类 · ${devices.length} 台`;
  container.innerHTML = `
    <button
      type="button"
      class="tree-node tree-root ${filter.dev_type === "all" ? "is-active" : ""}"
      data-runtime-tree-type="all"
      data-runtime-tree-name=""
    >
      <span>全部设备</span>
      <strong>${devices.length}</strong>
    </button>
    ${groupEntries.map(([devType, items]) => `
      <div class="tree-group">
        <button
          type="button"
          class="tree-node tree-type ${filter.dev_type === devType && !filter.dev_name ? "is-active" : filter.dev_type === devType ? "is-parent-active" : ""}"
          data-runtime-tree-type="${escapeHtml(devType)}"
          data-runtime-tree-name=""
        >
          <span>${escapeHtml(devType)}</span>
          <strong>${items.length}</strong>
        </button>
        <div class="tree-children">
          ${items.map((dev) => `
            <button
              type="button"
              class="tree-node tree-child ${filter.dev_type === dev.dev_type && filter.dev_name === dev.dev_name ? "is-active" : ""}"
              data-runtime-tree-type="${escapeHtml(dev.dev_type)}"
              data-runtime-tree-name="${escapeHtml(dev.dev_name)}"
            >
              <span>${escapeHtml(dev.dev_name)}</span>
              <small>${escapeHtml(deviceTreeBadge(dev))}</small>
            </button>
          `).join("")}
        </div>
      </div>
    `).join("")}
  `;
}

function setRuntimeDeviceFilter(devType, devName = "") {
  state.runtimeDeviceFilter = { dev_type: devType || "all", dev_name: devName || "" };
  renderRuntimeMonitor(true);
}

function runtimeFilterLabel(filter = state.runtimeDeviceFilter || { dev_type: "all", dev_name: "" }) {
  if (filter.dev_type === "all") return "全部设备";
  if (filter.dev_name) return filter.dev_name;
  return filter.dev_type;
}

function formatSetValues(setValues) {
  const entries = Object.entries(setValues || {});
  if (!entries.length) return "--";
  return entries.map(([key, value]) => `${key}=${value}`).join(" ");
}

function formatRuntimeSignal(value, unit) {
  const formatted = formatMeasurementValue(value);
  return formatted === "--" || !unit ? formatted : `${formatted} ${unit}`;
}

function renderRuntimeDeviceTable() {
  const container = $("deviceTable");
  if (!container) return;
  const devices = runtimeDevices();
  const rows = filteredRuntimeDevices(devices);
  $("runtimeDeviceSummary").textContent = `${runtimeFilterLabel()} · ${rows.length}/${devices.length} 台`;
  if (!devices.length) {
    container.innerHTML = '<div class="empty-state">暂无设备数据</div>';
    return;
  }
  if (!rows.length) {
    container.innerHTML = '<div class="empty-state">当前筛选无设备</div>';
    return;
  }
  container.innerHTML = `
    <table class="runtime-device-table">
      <thead>
        <tr>
          <th>设备</th>
          <th>类型</th>
          <th>投运</th>
          <th>状态</th>
          <th>模式</th>
          <th>设值</th>
          <th>控制指令</th>
          <th>实时值</th>
          <th>量测值</th>
        </tr>
      </thead>
      <tbody>
        ${rows.map((dev) => {
          const signal = runtimeDeviceTraceSignal(dev);
          return `
            <tr>
              <td>${escapeHtml(dev.dev_name)}</td>
              <td>${escapeHtml(dev.dev_type)}</td>
              <td><span class="status-dot ${dev.run_stat ? "on" : ""}"></span>${dev.run_stat ? "投入" : "退出"}</td>
              <td>${dev.status ? "闭合/可用" : "断开/故障"}</td>
              <td>${escapeHtml(dev.mode || "--")}</td>
              <td class="mono-cell">${escapeHtml(formatSetValues(dev.set_values))}</td>
              <td class="numeric-cell">${escapeHtml(formatRuntimeSignal(signal.control, signal.unit))}</td>
              <td class="numeric-cell">${escapeHtml(formatRuntimeSignal(signal.real, signal.unit))}</td>
              <td class="numeric-cell">${escapeHtml(formatRuntimeSignal(signal.scada, signal.unit))}</td>
            </tr>
          `;
        }).join("")}
      </tbody>
    </table>`;
}

function runtimeTraceDevicesForChart() {
  const rows = filteredRuntimeDevices();
  if (rows.length <= 1) return rows;
  const firstMeta = runtimeControlMeta(rows[0]);
  return rows.filter((dev) => {
    const meta = runtimeControlMeta(dev);
    return meta.kind === firstMeta.kind && meta.unit === firstMeta.unit;
  });
}

function runtimeTraceWindowPoints() {
  const history = state.runtimeTraceHistory || [];
  if (!history.length) return [];
  const range = runtimeTraceWindowRange();
  return history.filter((point) => point.minute >= range.startMinute && point.minute <= range.endMinute);
}

function runtimeTraceWindowRange() {
  const history = state.runtimeTraceHistory || [];
  const windowMinutes = Math.max(1, Number(state.runtimeTraceWindowMinutes) || 60);
  const fallbackMinute = Number(state.snapshot?.clock?.absolute_minute ?? state.snapshot?.clock?.minute ?? 0) || 0;
  const endMinute = history.length ? history[history.length - 1].minute : fallbackMinute;
  return { startMinute: endMinute - windowMinutes, endMinute, windowMinutes };
}

function runtimeFormatClockMinute(minute) {
  const total = ((Math.round(minute) % 1440) + 1440) % 1440;
  const hours = Math.floor(total / 60);
  const minutes = total % 60;
  return `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:00`;
}

function runtimeFormatWindowSpan(minutes) {
  if (minutes >= 60 && minutes % 60 === 0) return `${minutes / 60}小时`;
  return `${minutes}分钟`;
}

function runtimeAxisTickLabel(minute, range, index, lastIndex) {
  if (index === 0) return `-${runtimeFormatWindowSpan(range.windowMinutes)}`;
  if (index === lastIndex) return runtimeFormatClockMinute(range.endMinute);
  return runtimeFormatClockMinute(minute);
}

function runtimeTraceAxisTicks(range, canvasWidth) {
  const tickCount = canvasWidth < 480 ? 3 : canvasWidth < 760 ? 4 : 6;
  return Array.from({ length: tickCount }, (_unused, index) => {
    const ratio = index / Math.max(1, tickCount - 1);
    return range.startMinute + ratio * range.windowMinutes;
  });
}

function runtimeAggregateTracePoint(point, devices) {
  const keys = devices.map(deviceKey);
  const signals = keys.map((key) => point.devices[key]).filter(Boolean);
  const average = (field) => {
    const values = signals.map((signal) => numberOrNull(signal[field])).filter((value) => value !== null);
    if (!values.length) return null;
    return values.reduce((sum, value) => sum + value, 0) / values.length;
  };
  const first = signals[0] || {};
  return {
    minute: point.minute,
    sim_time: point.sim_time,
    control: average("control"),
    real: average("real"),
    scada: average("scada"),
    unit: first.unit || "",
    signal_kind: first.signal_kind || "",
  };
}

function resizeRuntimeTraceCanvas() {
  const canvas = $("runtimeTraceChart");
  if (!canvas) return false;
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(340, Math.round(rect.width || canvas.clientWidth || canvas.width));
  const height = Math.max(240, Math.round(rect.height || canvas.clientHeight || canvas.height));
  if (canvas.width === width && canvas.height === height) return false;
  canvas.width = width;
  canvas.height = height;
  return true;
}

function drawRuntimeTraceChart() {
  const canvas = $("runtimeTraceChart");
  if (!canvas) return;
  resizeRuntimeTraceCanvas();
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  const plot = width < 640
    ? { left: 42, right: 14, top: 28, bottom: 32 }
    : { left: 58, right: 24, top: 28, bottom: 36 };
  const left = plot.left;
  const right = width - plot.right;
  const top = plot.top;
  const bottom = height - plot.bottom;
  const chartDevices = runtimeTraceDevicesForChart();
  const range = runtimeTraceWindowRange();
  const points = runtimeTraceWindowPoints().map((point) => runtimeAggregateTracePoint(point, chartDevices));
  const values = points.flatMap((point) => [point.control, point.real, point.scada])
    .filter((value) => value !== null && Number.isFinite(value));
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#fcfeff";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "#d8e1e5";
  ctx.lineWidth = 1;
  ctx.font = "12px Microsoft YaHei, Arial";
  ctx.fillStyle = "#63717a";
  for (let i = 0; i <= 4; i += 1) {
    const y = top + i * ((bottom - top) / 4);
    ctx.beginPath();
    ctx.moveTo(left, y);
    ctx.lineTo(right, y);
    ctx.stroke();
  }
  const xTicks = runtimeTraceAxisTicks(range, width);
  xTicks.forEach((minute, tickIndex) => {
    const ratio = (minute - range.startMinute) / range.windowMinutes;
    const x = left + ratio * (right - left);
    ctx.strokeStyle = tickIndex === xTicks.length - 1 ? "#c9d6dc" : "#e7eef1";
    ctx.beginPath();
    ctx.moveTo(x, top);
    ctx.lineTo(x, bottom);
    ctx.stroke();
    ctx.fillStyle = "#63717a";
    ctx.textAlign = tickIndex === xTicks.length - 1 ? "right" : "left";
    const textOffset = tickIndex === 0 ? 0 : tickIndex === xTicks.length - 1 ? 0 : 4;
    ctx.fillText(runtimeAxisTickLabel(minute, range, tickIndex, xTicks.length - 1), x + textOffset, height - 12);
  });
  const label = runtimeFilterLabel();
  const chartLabel = chartDevices.length > 1 ? `${label} · ${chartDevices.length}台平均` : label;
  $("runtimeTraceSummary").textContent = `${chartLabel} · ${points.length} 点`;
  if (!chartDevices.length || !points.length || !values.length) {
    ctx.fillStyle = "#63717a";
    ctx.textAlign = "center";
    ctx.fillText("暂无跟踪数据", width / 2, height / 2);
    ctx.textAlign = "left";
    return;
  }
  let minValue = Math.min(...values);
  let maxValue = Math.max(...values);
  if (Math.abs(maxValue - minValue) < 1e-9) {
    minValue -= 1;
    maxValue += 1;
  }
  const padding = (maxValue - minValue) * 0.12;
  minValue -= padding;
  maxValue += padding;
  const xForMinute = (minute) => left + ((minute - range.startMinute) / range.windowMinutes) * (right - left);
  const yForValue = (value) => bottom - ((value - minValue) / (maxValue - minValue)) * (bottom - top);
  const drawSeries = (field, color, widthScale = 2) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = widthScale;
    ctx.beginPath();
    let started = false;
    points.forEach((point) => {
      const value = numberOrNull(point[field]);
      if (value === null) return;
      const x = xForMinute(point.minute);
      const y = yForValue(value);
      if (!started) {
        ctx.moveTo(x, y);
        started = true;
      } else {
        ctx.lineTo(x, y);
      }
    });
    if (started) ctx.stroke();
  };
  drawSeries("control", "#b87500", 2.5);
  drawSeries("real", "#008c8c", 2.5);
  drawSeries("scada", "#c93a3a", 2);
  ctx.fillStyle = "#63717a";
  ctx.textAlign = "left";
  ctx.fillText(formatMeasurementValue(maxValue), 8, top + 4);
  ctx.fillText(formatMeasurementValue(minValue), 8, bottom);
  const unit = points.find((point) => point.unit)?.unit || "";
  if (unit) ctx.fillText(unit, left, 18);
}

function renderRuntimeMonitor(force = false) {
  const activeEditor = document.activeElement?.closest?.("#runtimeTraceWindow");
  renderRuntimeDeviceTree();
  renderRuntimeDeviceTable();
  if (force || !activeEditor) drawRuntimeTraceChart();
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

function measurementUnit(measType) {
  const type = String(measType || "").toUpperCase();
  if (type.startsWith("P")) return "kW";
  if (type.startsWith("Q")) return "kvar";
  if (type.startsWith("V")) return "V";
  if (type.startsWith("I")) return "A";
  return "";
}

function appendMeasurementTrace(snapshot) {
  const clock = snapshot.clock || {};
  const result = snapshot.result || {};
  const summary = snapshot.summary || {};
  const signature = [
    snapshot.model?.id || state.activeModelId,
    clock.absolute_minute ?? clock.minute ?? "",
    clock.time || "",
    result.updated ?? "",
    result.solver_info || "",
    summary.scada_count ?? 0,
  ].join("|");
  if (signature === state.lastMeasurementTraceKey) return;
  state.lastMeasurementTraceKey = signature;
  const point = {
    minute: Number(clock.absolute_minute ?? clock.minute ?? state.measurementTraceHistory.length) || 0,
    sim_time: clock.time || "--",
    record_time: Date.now(),
    measurements: {},
  };
  measurementCompareRows(snapshot.measurements || {}).forEach((row) => {
    const key = measurementKey(row);
    point.measurements[key] = {
      name: row.name || "",
      dev_type: row.dev_type || "",
      dev_name: row.dev_name || "",
      meas_type: row.meas_type || "",
      unit: measurementUnit(row.meas_type),
      real: numberOrNull(row.real_value),
      scada: numberOrNull(row.scada_value),
      valid: Number(row.valid) === 1 ? 1 : 0,
    };
  });
  state.measurementTraceHistory.push(point);
  state.measurementTraceHistory = state.measurementTraceHistory.slice(-3000);
}

function ensureSelectedMeasurementKey(rows, allRows) {
  const availableRows = rows.length ? rows : allRows;
  const availableKeys = new Set(availableRows.map((row) => measurementKey(row)));
  if (state.selectedMeasurementKey && availableKeys.has(state.selectedMeasurementKey)) {
    return state.selectedMeasurementKey;
  }
  state.selectedMeasurementKey = availableRows.length ? measurementKey(availableRows[0]) : "";
  return state.selectedMeasurementKey;
}

function selectedMeasurementRow(rows = measurementCompareRows()) {
  if (!state.selectedMeasurementKey) return null;
  return rows.find((row) => measurementKey(row) === state.selectedMeasurementKey) || null;
}

function setSelectedMeasurementKey(key) {
  state.selectedMeasurementKey = key || "";
  renderMeasurementCompareTable();
  drawMeasurementTraceChart();
}

function measurementTraceWindowRange() {
  const history = state.measurementTraceHistory || [];
  const windowMinutes = Math.max(1, Number(state.measurementTraceWindowMinutes) || 60);
  const fallbackMinute = Number(state.snapshot?.clock?.absolute_minute ?? state.snapshot?.clock?.minute ?? 0) || 0;
  const endMinute = history.length ? history[history.length - 1].minute : fallbackMinute;
  return { startMinute: endMinute - windowMinutes, endMinute, windowMinutes };
}

function measurementTraceWindowPoints(key = state.selectedMeasurementKey) {
  if (!key) return [];
  const range = measurementTraceWindowRange();
  return (state.measurementTraceHistory || [])
    .filter((point) => point.minute >= range.startMinute && point.minute <= range.endMinute)
    .map((point) => {
      const measurement = point.measurements[key];
      if (!measurement) return null;
      return {
        minute: point.minute,
        sim_time: point.sim_time,
        real: measurement.real,
        scada: measurement.scada,
        unit: measurement.unit || "",
      };
    })
    .filter(Boolean);
}

function resizeMeasurementTraceCanvas() {
  const canvas = $("measurementTraceChart");
  if (!canvas) return false;
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(340, Math.round(rect.width || canvas.clientWidth || canvas.width));
  const height = Math.max(240, Math.round(rect.height || canvas.clientHeight || canvas.height));
  if (canvas.width === width && canvas.height === height) return false;
  canvas.width = width;
  canvas.height = height;
  return true;
}

function drawMeasurementTraceChart() {
  const canvas = $("measurementTraceChart");
  if (!canvas) return;
  resizeMeasurementTraceCanvas();
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  const plot = width < 640
    ? { left: 42, right: 14, top: 28, bottom: 32 }
    : { left: 58, right: 24, top: 28, bottom: 36 };
  const left = plot.left;
  const right = width - plot.right;
  const top = plot.top;
  const bottom = height - plot.bottom;
  const range = measurementTraceWindowRange();
  const allRows = measurementCompareRows();
  const selectedRow = selectedMeasurementRow(allRows);
  const points = measurementTraceWindowPoints();
  const values = points.flatMap((point) => [point.real, point.scada])
    .filter((value) => value !== null && Number.isFinite(value));
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#fcfeff";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "#d8e1e5";
  ctx.lineWidth = 1;
  ctx.font = "12px Microsoft YaHei, Arial";
  ctx.fillStyle = "#63717a";
  for (let i = 0; i <= 4; i += 1) {
    const y = top + i * ((bottom - top) / 4);
    ctx.beginPath();
    ctx.moveTo(left, y);
    ctx.lineTo(right, y);
    ctx.stroke();
  }
  const xTicks = runtimeTraceAxisTicks(range, width);
  xTicks.forEach((minute, tickIndex) => {
    const ratio = (minute - range.startMinute) / range.windowMinutes;
    const x = left + ratio * (right - left);
    ctx.strokeStyle = tickIndex === xTicks.length - 1 ? "#c9d6dc" : "#e7eef1";
    ctx.beginPath();
    ctx.moveTo(x, top);
    ctx.lineTo(x, bottom);
    ctx.stroke();
    ctx.fillStyle = "#63717a";
    ctx.textAlign = tickIndex === xTicks.length - 1 ? "right" : "left";
    const textOffset = tickIndex === 0 ? 0 : tickIndex === xTicks.length - 1 ? 0 : 4;
    ctx.fillText(runtimeAxisTickLabel(minute, range, tickIndex, xTicks.length - 1), x + textOffset, height - 12);
  });
  const label = selectedRow?.name || "请选择测点";
  $("measurementTraceSummary").textContent = `${label} · ${points.length} 点`;
  if (!selectedRow || !points.length || !values.length) {
    ctx.fillStyle = "#63717a";
    ctx.textAlign = "center";
    ctx.fillText("暂无跟踪数据", width / 2, height / 2);
    ctx.textAlign = "left";
    return;
  }
  let minValue = Math.min(...values);
  let maxValue = Math.max(...values);
  if (Math.abs(maxValue - minValue) < 1e-9) {
    minValue -= 1;
    maxValue += 1;
  }
  const padding = (maxValue - minValue) * 0.12;
  minValue -= padding;
  maxValue += padding;
  const xForMinute = (minute) => left + ((minute - range.startMinute) / range.windowMinutes) * (right - left);
  const yForValue = (value) => bottom - ((value - minValue) / (maxValue - minValue)) * (bottom - top);
  const drawSeries = (field, color, widthScale = 2.5) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = widthScale;
    ctx.beginPath();
    let started = false;
    points.forEach((point) => {
      const value = numberOrNull(point[field]);
      if (value === null) return;
      const x = xForMinute(point.minute);
      const y = yForValue(value);
      if (!started) {
        ctx.moveTo(x, y);
        started = true;
      } else {
        ctx.lineTo(x, y);
      }
    });
    if (started) ctx.stroke();
  };
  drawSeries("real", "#008c8c", 2.5);
  drawSeries("scada", "#c93a3a", 2);
  ctx.fillStyle = "#63717a";
  ctx.textAlign = "left";
  ctx.fillText(formatMeasurementValue(maxValue), 8, top + 4);
  ctx.fillText(formatMeasurementValue(minValue), 8, bottom);
  const unit = points.find((point) => point.unit)?.unit || measurementUnit(selectedRow.meas_type);
  if (unit) ctx.fillText(unit, left, 18);
}

function measurementCompareDevices(rows = measurementCompareRows()) {
  const devices = new Map();
  rows.forEach((row) => {
    if (!row.dev_type || !row.dev_name) return;
    const key = deviceKey(row);
    const entry = devices.get(key) || { dev_type: row.dev_type, dev_name: row.dev_name, count: 0 };
    entry.count += 1;
    devices.set(key, entry);
  });
  return Array.from(devices.values()).sort((left, right) => {
    const typeCompare = String(left.dev_type).localeCompare(String(right.dev_type));
    return typeCompare || String(left.dev_name).localeCompare(String(right.dev_name));
  });
}

function filteredMeasurementCompareRows(rows = measurementCompareRows()) {
  const filter = state.measurementCompareFilter || { dev_type: "all", dev_name: "" };
  return rows.filter((row) => {
    if (filter.dev_type && filter.dev_type !== "all" && row.dev_type !== filter.dev_type) return false;
    if (filter.dev_name && row.dev_name !== filter.dev_name) return false;
    return true;
  });
}

function renderMeasurementCompareDeviceTree(rows = measurementCompareRows()) {
  const container = $("measurementCompareDeviceTree");
  if (!container) return;
  const devices = measurementCompareDevices(rows);
  const filter = state.measurementCompareFilter || { dev_type: "all", dev_name: "" };
  const groupEntries = groupedByDeviceType(devices);
  $("measurementCompareTreeSummary").textContent = `${groupEntries.length} 类 · ${devices.length} 台`;
  container.innerHTML = `
    <button
      type="button"
      class="tree-node tree-root ${filter.dev_type === "all" ? "is-active" : ""}"
      data-measurement-tree-type="all"
      data-measurement-tree-name=""
    >
      <span>全部设备</span>
      <strong>${devices.length}</strong>
    </button>
    ${groupEntries.map(([devType, items]) => `
      <div class="tree-group">
        <button
          type="button"
          class="tree-node tree-type ${filter.dev_type === devType && !filter.dev_name ? "is-active" : filter.dev_type === devType ? "is-parent-active" : ""}"
          data-measurement-tree-type="${escapeHtml(devType)}"
          data-measurement-tree-name=""
        >
          <span>${escapeHtml(devType)}</span>
          <strong>${items.length}</strong>
        </button>
        <div class="tree-children">
          ${items.map((item) => `
            <button
              type="button"
              class="tree-node tree-child ${filter.dev_type === item.dev_type && filter.dev_name === item.dev_name ? "is-active" : ""}"
              data-measurement-tree-type="${escapeHtml(item.dev_type)}"
              data-measurement-tree-name="${escapeHtml(item.dev_name)}"
            >
              <span>${escapeHtml(item.dev_name)}</span>
              <small>${escapeHtml(item.count)}点</small>
            </button>
          `).join("")}
        </div>
      </div>
    `).join("")}
  `;
}

function setMeasurementCompareFilter(devType, devName = "") {
  state.measurementCompareFilter = { dev_type: devType || "all", dev_name: devName || "" };
  renderMeasurementCompareTable();
}

function renderMeasurementCompareTable() {
  const container = $("measurementCompareTable");
  if (!container) return;
  const allRows = measurementCompareRows();
  renderMeasurementCompareDeviceTree(allRows);
  const rows = filteredMeasurementCompareRows(allRows);
  const selectedKey = ensureSelectedMeasurementKey(rows, allRows);
  const validCount = rows.filter((row) => Number(row.valid) === 1).length;
  $("measurementCompareSummary").textContent = `${rows.length}/${allRows.length} 点 · 有效 ${validCount} 点`;
  if (!allRows.length) {
    container.innerHTML = '<div class="empty-state">暂无实时量测数据</div>';
    drawMeasurementTraceChart();
    return;
  }
  if (!rows.length) {
    container.innerHTML = '<div class="empty-state">当前筛选无量测</div>';
    drawMeasurementTraceChart();
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
          const key = measurementKey(row);
          return `
            <tr
              class="${key === selectedKey ? "is-selected" : ""}"
              data-measurement-select-key="${escapeHtml(key)}"
              tabindex="0"
              aria-selected="${key === selectedKey ? "true" : "false"}"
            >
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
  drawMeasurementTraceChart();
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

function deviceTreeBadge(dev) {
  const raw = dev.raw || {};
  return String(dev.mode || raw.control_type || raw.ctrl_mode || (Number(dev.run_stat ?? 1) !== 0 ? "投" : "退"));
}

function groupedByDeviceType(items) {
  const groups = new Map();
  items.forEach((item) => {
    const devType = item.dev_type || "未分类";
    const list = groups.get(devType) || [];
    list.push(item);
    groups.set(devType, list);
  });
  return Array.from(groups.entries())
    .map(([devType, list]) => [
      devType,
      list.sort((left, right) => String(left.dev_name || left.name || "").localeCompare(String(right.dev_name || right.name || ""))),
    ])
    .sort(([left], [right]) => String(left).localeCompare(String(right)));
}

function filteredFaultDevices() {
  const filter = state.faultDeviceFilter || { dev_type: "all", dev_name: "" };
  return faultDevices()
    .map((dev, index) => ({ dev, index }))
    .filter(({ dev }) => {
      if (filter.dev_type && filter.dev_type !== "all" && dev.dev_type !== filter.dev_type) return false;
      if (filter.dev_name && dev.dev_name !== filter.dev_name) return false;
      return true;
    });
}

function filteredFaultMeasurements() {
  const filter = state.faultMeasurementFilter || { dev_type: "all", dev_name: "", key: "" };
  return faultMeasurements()
    .map((meas, index) => ({ meas, index }))
    .filter(({ meas }) => {
      if (filter.dev_type && filter.dev_type !== "all" && meas.dev_type !== filter.dev_type) return false;
      if (filter.dev_name && meas.dev_name !== filter.dev_name) return false;
      if (filter.key && measurementKey(meas) !== filter.key) return false;
      return true;
    });
}

function faultMeasurementDevices(measurements = faultMeasurements()) {
  const devices = new Map();
  measurements.forEach((meas) => {
    if (!meas.dev_type || !meas.dev_name) return;
    const key = deviceKey(meas);
    const entry = devices.get(key) || { dev_type: meas.dev_type, dev_name: meas.dev_name, count: 0 };
    entry.count += 1;
    devices.set(key, entry);
  });
  return Array.from(devices.values()).sort((left, right) => {
    const typeCompare = String(left.dev_type).localeCompare(String(right.dev_type));
    return typeCompare || String(left.dev_name).localeCompare(String(right.dev_name));
  });
}

function renderFaultDeviceTree() {
  const container = $("faultDeviceTree");
  if (!container) return;
  const devices = faultDevices();
  const filter = state.faultDeviceFilter || { dev_type: "all", dev_name: "" };
  const groupEntries = groupedByDeviceType(devices);
  $("faultDeviceTreeSummary").textContent = `${groupEntries.length} 类 · ${devices.length} 台`;
  container.innerHTML = `
    <button
      type="button"
      class="tree-node tree-root ${filter.dev_type === "all" ? "is-active" : ""}"
      data-fault-device-tree-type="all"
      data-fault-device-tree-name=""
    >
      <span>全部设备</span>
      <strong>${devices.length}</strong>
    </button>
    ${groupEntries.map(([devType, items]) => `
      <div class="tree-group">
        <button
          type="button"
          class="tree-node tree-type ${filter.dev_type === devType && !filter.dev_name ? "is-active" : filter.dev_type === devType ? "is-parent-active" : ""}"
          data-fault-device-tree-type="${escapeHtml(devType)}"
          data-fault-device-tree-name=""
        >
          <span>${escapeHtml(devType)}</span>
          <strong>${items.length}</strong>
        </button>
        <div class="tree-children">
          ${items.map((dev) => `
            <button
              type="button"
              class="tree-node tree-child ${filter.dev_type === dev.dev_type && filter.dev_name === dev.dev_name ? "is-active" : ""}"
              data-fault-device-tree-type="${escapeHtml(dev.dev_type)}"
              data-fault-device-tree-name="${escapeHtml(dev.dev_name)}"
            >
              <span>${escapeHtml(dev.dev_name)}</span>
              <small>${escapeHtml(deviceTreeBadge(dev))}</small>
            </button>
          `).join("")}
        </div>
      </div>
    `).join("")}
  `;
}

function renderFaultMeasurementTree() {
  const container = $("faultMeasurementTree");
  if (!container) return;
  const measurements = faultMeasurements();
  const devices = faultMeasurementDevices(measurements);
  const filter = state.faultMeasurementFilter || { dev_type: "all", dev_name: "", key: "" };
  const groupEntries = groupedByDeviceType(devices);
  $("faultMeasurementTreeSummary").textContent = `${groupEntries.length} 类 · ${devices.length} 台`;
  container.innerHTML = `
    <button
      type="button"
      class="tree-node tree-root ${filter.dev_type === "all" ? "is-active" : ""}"
      data-fault-measurement-tree-type="all"
      data-fault-measurement-tree-name=""
    >
      <span>全部设备</span>
      <strong>${devices.length}</strong>
    </button>
    ${groupEntries.map(([devType, items]) => `
      <div class="tree-group">
        <button
          type="button"
          class="tree-node tree-type ${filter.dev_type === devType && !filter.dev_name ? "is-active" : filter.dev_type === devType ? "is-parent-active" : ""}"
          data-fault-measurement-tree-type="${escapeHtml(devType)}"
          data-fault-measurement-tree-name=""
        >
          <span>${escapeHtml(devType)}</span>
          <strong>${items.length}</strong>
        </button>
        <div class="tree-children">
          ${items.map((dev) => `
            <button
              type="button"
              class="tree-node tree-child ${filter.dev_type === dev.dev_type && filter.dev_name === dev.dev_name ? "is-active" : ""}"
              data-fault-measurement-tree-type="${escapeHtml(dev.dev_type)}"
              data-fault-measurement-tree-name="${escapeHtml(dev.dev_name)}"
            >
              <span>${escapeHtml(dev.dev_name)}</span>
              <small>${escapeHtml(dev.count)}点</small>
            </button>
          `).join("")}
        </div>
      </div>
    `).join("")}
  `;
}

function setDeviceFaultFilter(devType, devName = "") {
  state.faultDeviceFilter = { dev_type: devType || "all", dev_name: devName || "" };
  renderFaults(true);
}

function setMeasurementFaultFilter(devType, devName = "") {
  state.faultMeasurementFilter = { dev_type: devType || "all", dev_name: devName || "", key: "" };
  renderFaults(true);
}

function renderFaults(force = false) {
  const activeEditor = document.activeElement?.closest?.("#deviceFaultTable, #measurementFaultTable");
  if (!force && activeEditor) return;
  renderFaultDeviceTree();
  renderDeviceFaultTable();
  renderFaultMeasurementTree();
  renderMeasurementFaultTable();
}

function renderDeviceFaultTable() {
  const container = $("deviceFaultTable");
  const devices = faultDevices();
  const rows = filteredFaultDevices();
  if (!container) return;
  $("deviceFaultSummary").textContent = `${state.deviceFaults.length} 个故障 · 显示 ${rows.length}/${devices.length} 台`;
  if (!devices.length) {
    container.innerHTML = '<div class="empty-state">暂无设备数据</div>';
    return;
  }
  if (!rows.length) {
    container.innerHTML = '<div class="empty-state">当前筛选无设备</div>';
    return;
  }
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
        ${rows.map(({ dev, index }) => {
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
  const rows = filteredFaultMeasurements();
  if (!container) return;
  $("measurementFaultSummary").textContent = `${state.measurementFaults.length} 个故障 · 显示 ${rows.length}/${measurements.length} 点`;
  if (!measurements.length) {
    container.innerHTML = '<div class="empty-state">暂无量测数据</div>';
    return;
  }
  if (!rows.length) {
    container.innerHTML = '<div class="empty-state">当前筛选无量测</div>';
    return;
  }
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
        ${rows.map(({ meas, index }) => {
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
  const config = curveModeConfig();
  await api("/api/curves", {
    method: "POST",
    body: JSON.stringify({
      mode: state.curveMode,
      point_count: config.pointCount,
      time_step_minutes: config.stepMinutes,
      weather: state.weatherPoints,
      loads: state.loadPointsByName,
    }),
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
document.querySelectorAll("[data-curve-mode]").forEach((button) => {
  button.addEventListener("click", () => setCurveMode(button.dataset.curveMode));
});
$("modelSelector").addEventListener("change", (event) => setActiveModel(event.target.value));
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
  const curveTreeButton = event.target.closest("[data-curve-tree-type]");
  if (curveTreeButton) {
    if (curveTreeButton.dataset.curveFamily) {
      selectCurveFamily(curveTreeButton.dataset.curveFamily);
    } else {
      toggleCurveSelection(curveTreeButton.dataset.curveKey);
    }
  }
  const faultDeviceTreeButton = event.target.closest("[data-fault-device-tree-type]");
  if (faultDeviceTreeButton) {
    setDeviceFaultFilter(
      faultDeviceTreeButton.dataset.faultDeviceTreeType,
      faultDeviceTreeButton.dataset.faultDeviceTreeName || "",
    );
  }
  const faultMeasurementTreeButton = event.target.closest("[data-fault-measurement-tree-type]");
  if (faultMeasurementTreeButton) {
    setMeasurementFaultFilter(
      faultMeasurementTreeButton.dataset.faultMeasurementTreeType,
      faultMeasurementTreeButton.dataset.faultMeasurementTreeName || "",
    );
  }
  const measurementSelectRow = event.target.closest("[data-measurement-select-key]");
  if (measurementSelectRow) {
    setSelectedMeasurementKey(measurementSelectRow.dataset.measurementSelectKey || "");
  }
  const measurementTreeButton = event.target.closest("[data-measurement-tree-type]");
  if (measurementTreeButton) {
    setMeasurementCompareFilter(
      measurementTreeButton.dataset.measurementTreeType,
      measurementTreeButton.dataset.measurementTreeName || "",
    );
  }
  const runtimeTreeButton = event.target.closest("[data-runtime-tree-type]");
  if (runtimeTreeButton) {
    setRuntimeDeviceFilter(
      runtimeTreeButton.dataset.runtimeTreeType,
      runtimeTreeButton.dataset.runtimeTreeName || "",
    );
  }
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
initRuntimeMonitor();
initMeasurementMonitor();
setFaultTab(state.activeFaultTab);
renderFaults(true);
setInterval(refresh, 2000);
loadModels().finally(refresh);
