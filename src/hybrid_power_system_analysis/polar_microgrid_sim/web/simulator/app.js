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
  curveCursor: { visible: false, x: 0, y: 0, index: 0 },
  settingsLoaded: false,
  activeFaultTab: "devices",
  faultDeviceFilter: { dev_type: "all", dev_name: "" },
  faultMeasurementFilter: { dev_type: "all", dev_name: "", key: "" },
  modelDeviceFilter: { dev_type: "all", dev_name: "" },
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
  collapsedDeviceTreeGroups: {},
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

function isDeviceTreeGroupCollapsed(scope, groupKey) {
  return Boolean(state.collapsedDeviceTreeGroups?.[scope]?.[groupKey]);
}

function toggleDeviceTreeGroup(scope, groupKey) {
  if (!scope || !groupKey || groupKey === "all") return;
  if (!state.collapsedDeviceTreeGroups[scope]) {
    state.collapsedDeviceTreeGroups[scope] = {};
  }
  if (state.collapsedDeviceTreeGroups[scope][groupKey]) {
    delete state.collapsedDeviceTreeGroups[scope][groupKey];
  } else {
    state.collapsedDeviceTreeGroups[scope][groupKey] = true;
  }
}

function deviceTreeTypeAttrs(scope, groupKey, isCollapsed) {
  return `
          data-tree-toggle-scope="${escapeHtml(scope)}"
          data-tree-toggle-group="${escapeHtml(groupKey)}"
          aria-expanded="${isCollapsed ? "false" : "true"}"`;
}

function deviceTreeTypeLabel(label) {
  return `
          <span class="tree-title">
            <i class="tree-toggle" aria-hidden="true"></i>
            <span class="tree-title-text">${escapeHtml(label)}</span>
          </span>`;
}

function deviceTreeChildren(isCollapsed, childrenHtml) {
  if (isCollapsed) return "";
  return `
        <div class="tree-children">
          ${childrenHtml}
        </div>`;
}

function renderClock(clock) {
  if (!clock) return;
  $("simState").textContent = clock.state || "stopped";
  $("simTime").textContent = clock.time || "00:00:00";
  $("simSpeed").textContent = `x${clock.speed ?? 1}`;
  const readout = document.querySelector(".clock-readout");
  if (readout) {
    readout.dataset.clockState = clock.state || "stopped";
  }
  document.querySelectorAll("[data-clock]").forEach((button) => {
    const action = button.dataset.clock;
    const isActive =
      (action === "start" && clock.state === "running") ||
      (action === "pause" && clock.state === "paused") ||
      (action === "stop" && clock.state === "stopped");
    button.classList.toggle("is-active", isActive);
    if (["start", "pause", "stop"].includes(action)) {
      button.setAttribute("aria-pressed", isActive ? "true" : "false");
    }
  });
}

function setClockButtonsBusy(isBusy) {
  document.querySelectorAll("[data-clock]").forEach((button) => {
    button.disabled = isBusy;
    button.classList.toggle("is-busy", isBusy);
  });
}

async function controlClock(action) {
  setClockButtonsBusy(true);
  try {
    const clock = await api("/api/clock", { method: "POST", body: JSON.stringify({ action }) });
    renderClock(clock);
    await refresh();
  } catch (error) {
    $("simState").textContent = "error";
    $("solverInfo").textContent = "时钟控制失败";
    throw error;
  } finally {
    setClockButtonsBusy(false);
  }
}

function setCloneModelMessage(text, kind = "") {
  const message = $("cloneModelMessage");
  if (!message) return;
  message.textContent = text || "";
  message.classList.toggle("is-error", kind === "error");
  message.classList.toggle("is-ok", kind === "ok");
}

function setCloneConfirmEnabled(isEnabled) {
  const confirm = $("confirmCloneModel");
  if (confirm) confirm.disabled = !isEnabled;
}

function validateCloneModelName(showBlank = false) {
  const input = $("cloneModelName");
  const name = String(input?.value || "").trim();
  if (!name) {
    setCloneConfirmEnabled(false);
    setCloneModelMessage(showBlank ? "请输入新模型名称。" : "", showBlank ? "error" : "");
    return false;
  }
  if (isModelNameTaken(name)) {
    setCloneConfirmEnabled(false);
    setCloneModelMessage(`模型已存在：${name}`, "error");
    return false;
  }
  setCloneConfirmEnabled(true);
  setCloneModelMessage("");
  return true;
}

function openCloneModelDialog() {
  const dialog = $("cloneModelDialog");
  const input = $("cloneModelName");
  if (!dialog || !input) return;
  input.value = modelCloneDefaultName();
  validateCloneModelName();
  dialog.hidden = false;
  requestAnimationFrame(() => {
    input.focus();
    input.select();
  });
}

function closeCloneModelDialog() {
  const dialog = $("cloneModelDialog");
  if (dialog) dialog.hidden = true;
}

function apiErrorText(error) {
  try {
    return JSON.parse(error.message)?.error || error.message;
  } catch (_parseError) {
    return error.message || "操作失败";
  }
}

function modelKey(value) {
  const text = String(value ?? "").trim();
  const cleaned = Array.from(text).map((char) => (
    /[\p{L}\p{N}_-]/u.test(char) ? char : "_"
  )).join("").replace(/^_+|_+$/g, "");
  return (cleaned || "default").toLocaleLowerCase();
}

function normalizeModels(models) {
  const seen = new Set();
  const unique = [];
  (models || []).forEach((model) => {
    const keys = [modelKey(model.id), modelKey(model.name || model.id)];
    if (keys.some((key) => seen.has(key))) return;
    keys.forEach((key) => seen.add(key));
    unique.push(model);
  });
  return unique;
}

function isModelNameTaken(name) {
  const key = modelKey(name);
  return normalizeModels(state.models).some((model) => (
    modelKey(model.id) === key || modelKey(model.name || model.id) === key
  ));
}

function uniqueCloneName(baseName) {
  const base = String(baseName || "model").trim().replace(/\s+/g, "_") || "model";
  const taken = new Set();
  normalizeModels(state.models).forEach((model) => {
    taken.add(modelKey(model.id));
    taken.add(modelKey(model.name || model.id));
  });
  const first = `${base}_copy`;
  if (!taken.has(modelKey(first))) return first;
  for (let index = 2; index < 1000; index += 1) {
    const candidate = `${base}_copy_${index}`;
    if (!taken.has(modelKey(candidate))) return candidate;
  }
  return `${base}_copy_${Date.now()}`;
}

function modelCloneDefaultName() {
  const active = state.models.find((model) => model.id === state.activeModelId) || state.models[0] || {};
  const base = String(active.name || active.id || "model").replace(/\s+/g, "_");
  return uniqueCloneName(base);
}

function setCloneModelBusy(isBusy) {
  const confirm = $("confirmCloneModel");
  const button = $("cloneModelButton");
  const input = $("cloneModelName");
  if (confirm) {
    confirm.disabled = isBusy;
    confirm.textContent = isBusy ? "复制中" : "复制";
  }
  if (button) button.disabled = isBusy;
  if (input) input.disabled = isBusy;
}

async function cloneCurrentModel() {
  const input = $("cloneModelName");
  const name = String(input?.value || "").trim();
  if (!validateCloneModelName(true)) {
    input?.focus();
    return;
  }
  setCloneModelBusy(true);
  setCloneModelMessage("正在复制模型文件夹...");
  try {
    const result = await api("/api/models/clone", {
      method: "POST",
      body: JSON.stringify({ name }),
    });
    state.models = normalizeModels(Array.isArray(result.models) ? result.models : []);
    const newModelId = result.model?.id || result.active_model_id || name;
    closeCloneModelDialog();
    setActiveModel(newModelId, true);
  } catch (error) {
    setCloneModelMessage(apiErrorText(error), "error");
  } finally {
    setCloneModelBusy(false);
    if (!$("cloneModelDialog").hidden) validateCloneModelName();
  }
}

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
  if (target === "model") {
    requestAnimationFrame(() => renderGridModelPage());
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
  state.models = normalizeModels(state.models);
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
  state.modelDeviceFilter = { dev_type: "all", dev_name: "" };
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
    state.models = normalizeModels(Array.isArray(catalog.models) ? catalog.models : []);
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
  drawCurveCursor(ctx, canvas, plot, metas);
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

function curveXFromPointIndex(index, canvas) {
  const plot = curvePlot(canvas);
  const left = plot.left;
  const right = canvas.width - plot.right;
  return left + (clamp(index, 0, curvePointCount() - 1) / Math.max(1, curvePointCount() - 1)) * (right - left);
}

function setCurveCursorFromEvent(event, shouldDraw = true) {
  const canvas = $("curveEditorChart");
  if (!canvas) return;
  const pos = pointerPositionOnCanvas(event);
  const plot = curvePlot(canvas);
  const left = plot.left;
  const right = canvas.width - plot.right;
  const top = plot.top;
  const bottom = canvas.height - plot.bottom;
  if (pos.x < left || pos.x > right || pos.y < top || pos.y > bottom) {
    state.curveCursor = { visible: false, x: pos.x, y: pos.y, index: state.curveCursor.index || 0 };
    if (shouldDraw) drawCurves();
    return;
  }
  state.curveCursor = {
    visible: true,
    x: clamp(pos.x, left, right),
    y: clamp(pos.y, top, bottom),
    index: curvePointIndexFromX(pos.x, canvas),
  };
  if (shouldDraw) drawCurves();
}

function hideCurveCursor() {
  if (!state.curveCursor.visible) return;
  state.curveCursor.visible = false;
  drawCurves();
}

function drawCurveTooltipBox(ctx, x, y, width, height, radius = 8) {
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.lineTo(x + width - radius, y);
  ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
  ctx.lineTo(x + width, y + height - radius);
  ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
  ctx.lineTo(x + radius, y + height);
  ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
  ctx.lineTo(x, y + radius);
  ctx.quadraticCurveTo(x, y, x + radius, y);
  ctx.closePath();
}

function drawCurveCursor(ctx, canvas, plot, metas) {
  const cursor = state.curveCursor;
  if (!cursor.visible || !metas.length) return;
  const left = plot.left;
  const right = canvas.width - plot.right;
  const top = plot.top;
  const bottom = canvas.height - plot.bottom;
  const index = clamp(cursor.index, 0, curvePointCount() - 1);
  const x = curveXFromPointIndex(index, canvas);
  const y = clamp(cursor.y, top, bottom);
  const tooltipMetas = metas.slice(0, 6);
  const extraCount = Math.max(0, metas.length - tooltipMetas.length);
  const timeLabel = formatCurveTableTime(pointMinute(index));
  const valueLines = tooltipMetas.map((meta) => {
    const value = roundCurveValue(meta.key, state.curveSeries[meta.key]?.[index] ?? 0);
    return { meta, text: `${meta.label}: ${value} ${meta.unit}` };
  });

  ctx.save();
  ctx.strokeStyle = "rgba(29, 57, 66, 0.58)";
  ctx.lineWidth = 1;
  ctx.setLineDash([5, 4]);
  ctx.beginPath();
  ctx.moveTo(x, top);
  ctx.lineTo(x, bottom);
  ctx.moveTo(left, y);
  ctx.lineTo(right, y);
  ctx.stroke();
  ctx.setLineDash([]);

  tooltipMetas.forEach((meta) => {
    const values = state.curveSeries[meta.key] || [];
    if (!values.length) return;
    const markerY = valueToY(values[index], meta, canvas);
    ctx.fillStyle = "#ffffff";
    ctx.strokeStyle = meta.color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(x, markerY, 4.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  });

  ctx.font = "12px Microsoft YaHei, Arial";
  const title = `时刻: ${timeLabel}`;
  const point = `点号: ${index + 1}`;
  const lineTexts = [title, point, ...valueLines.map((line) => line.text), extraCount ? `另有 ${extraCount} 条曲线` : ""].filter(Boolean);
  const tooltipWidth = Math.max(154, ...lineTexts.map((line) => ctx.measureText(line).width + 28));
  const lineHeight = 18;
  const tooltipHeight = 16 + lineTexts.length * lineHeight;
  let tooltipX = x + 14;
  let tooltipY = y + 14;
  if (tooltipX + tooltipWidth > right - 6) tooltipX = x - tooltipWidth - 14;
  if (tooltipY + tooltipHeight > bottom - 6) tooltipY = y - tooltipHeight - 14;
  tooltipX = clamp(tooltipX, left + 6, right - tooltipWidth - 6);
  tooltipY = clamp(tooltipY, top + 6, bottom - tooltipHeight - 6);

  ctx.shadowColor = "rgba(28, 45, 52, 0.18)";
  ctx.shadowBlur = 12;
  ctx.shadowOffsetY = 3;
  ctx.fillStyle = "rgba(255, 255, 255, 0.96)";
  drawCurveTooltipBox(ctx, tooltipX, tooltipY, tooltipWidth, tooltipHeight);
  ctx.fill();
  ctx.shadowColor = "transparent";
  ctx.strokeStyle = "rgba(171, 190, 198, 0.9)";
  ctx.stroke();

  ctx.fillStyle = "#1f3037";
  ctx.font = "700 12px Microsoft YaHei, Arial";
  ctx.fillText(title, tooltipX + 10, tooltipY + 18);
  ctx.font = "12px Microsoft YaHei, Arial";
  ctx.fillStyle = "#63717a";
  ctx.fillText(point, tooltipX + 10, tooltipY + 36);
  valueLines.forEach((line, lineIndex) => {
    const textY = tooltipY + 54 + lineIndex * lineHeight;
    ctx.fillStyle = line.meta.color;
    ctx.fillRect(tooltipX + 10, textY - 7, 10, 3);
    ctx.fillStyle = "#314850";
    ctx.fillText(line.text, tooltipX + 26, textY);
  });
  if (extraCount) {
    ctx.fillStyle = "#63717a";
    ctx.fillText(`另有 ${extraCount} 条曲线`, tooltipX + 10, tooltipY + 54 + valueLines.length * lineHeight);
  }
  ctx.restore();
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
    setCurveCursorFromEvent(event, false);
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
    setCurveCursorFromEvent(event, !state.isCurveDragging);
    if (state.isCurveDragging) {
      event.preventDefault();
      applyCurveDrag(event);
    }
  });
  canvas.addEventListener("pointerleave", () => {
    if (!state.isCurveDragging) hideCurveCursor();
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
  renderClock(snapshot.clock);
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
  renderGridModelPage();
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

function gridModelDevices() {
  return state.snapshot?.devices || [];
}

function gridModelFilterMatches(dev, filter = state.modelDeviceFilter || { dev_type: "all", dev_name: "" }) {
  if (filter.dev_type && filter.dev_type !== "all" && dev.dev_type !== filter.dev_type) return false;
  if (filter.dev_name && dev.dev_name !== filter.dev_name) return false;
  return true;
}

function filteredGridModelDevices(devices = gridModelDevices()) {
  return devices.filter((dev) => gridModelFilterMatches(dev));
}

function gridModelFilterLabel(filter = state.modelDeviceFilter || { dev_type: "all", dev_name: "" }) {
  if (filter.dev_type === "all") return "全部设备";
  if (filter.dev_name) return filter.dev_name;
  return filter.dev_type;
}

function formatModelParamValue(value) {
  if (value === null || value === undefined || value === "") return "--";
  if (Array.isArray(value)) return value.join(", ");
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function modelDeviceIndexValue(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : Number.POSITIVE_INFINITY;
}

function compareModelRowsByIndex(left, right) {
  const indexCompare = modelDeviceIndexValue(left.idx ?? left.raw?.idx) - modelDeviceIndexValue(right.idx ?? right.raw?.idx);
  if (indexCompare) return indexCompare;
  return String(left.name || left.dev_name || "").localeCompare(String(right.name || right.dev_name || ""));
}

function modelAttributeRecordForDevice(dev) {
  const record = {
    dev_type: dev.dev_type || "--",
    dev_name: dev.dev_name || "--",
    idx: formatModelParamValue(dev.idx ?? dev.raw?.idx),
    name: formatModelParamValue(dev.dev_name || dev.raw?.name),
  };
  Object.entries(dev.raw || {}).forEach(([key, value]) => {
    if (["idx", "name", "dev_name", "dev_type"].includes(key)) return;
    record[key] = formatModelParamValue(value);
  });
  record.run_stat = formatModelParamValue(dev.run_stat ?? record.run_stat);
  record.status = formatModelParamValue(dev.status ?? record.status);
  record.mode = formatModelParamValue(dev.mode || dev.raw?.control_type || dev.raw?.ctrl_mode || record.mode);
  if ((dev.set_types || []).length) record.set_types = formatModelParamValue(dev.set_types);
  Object.entries(dev.set_values || {}).forEach(([key, value]) => {
    record[key] = formatModelParamValue(value);
  });
  return record;
}

function modelAttributeLabel(key) {
  const labels = {
    idx: "idx",
    name: "名称",
  };
  return labels[key] || key;
}

function modelAttributeColumns(records) {
  const fixed = ["idx", "name"];
  const preferred = [
    "node",
    "from_node",
    "to_node",
    "ac_node",
    "dc_node",
    "control_type",
    "ctrl_mode",
    "mode",
    "run_stat",
    "status",
    "p_set",
    "q_set",
    "v_set",
    "p_ac_set",
    "q_ac_set",
    "v_ac_set",
    "p_dc_set",
    "v_dc_set",
    "pv0",
    "pv1",
    "pv2",
    "qv0",
    "qv1",
    "qv2",
    "pbase",
    "qbase",
    "pmax",
    "pmin",
    "qmax",
    "qmin",
    "soc_curr",
    "alpha",
    "set_types",
  ];
  const seen = new Set([...fixed, "dev_type", "dev_name"]);
  const keys = [];
  const appendKey = (key) => {
    if (!key || seen.has(key)) return;
    if (!records.some((record) => record[key] !== undefined && record[key] !== "--")) return;
    seen.add(key);
    keys.push(key);
  };
  preferred.forEach(appendKey);
  records.forEach((record) => {
    Object.keys(record).forEach(appendKey);
  });
  return [...fixed, ...keys].map((key) => ({ key, label: modelAttributeLabel(key) }));
}

function groupedModelAttributeRecords(records) {
  const groups = new Map();
  records.forEach((record) => {
    const devType = record.dev_type || "未分类";
    const rows = groups.get(devType) || [];
    rows.push(record);
    groups.set(devType, rows);
  });
  return Array.from(groups.entries())
    .map(([devType, rows]) => [
      devType,
      rows.sort(compareModelRowsByIndex),
    ])
    .sort(([left], [right]) => String(left).localeCompare(String(right)));
}

function renderModelAttributeTable(rows) {
  const columns = modelAttributeColumns(rows);
  return `
    <table class="model-param-table">
      <thead>
        <tr>
          ${columns.map((column) => `<th>${escapeHtml(column.label)}</th>`).join("")}
        </tr>
      </thead>
      <tbody>
        ${rows.map((row) => `
          <tr>
            ${columns.map((column) => `<td class="attr-value">${escapeHtml(row[column.key] ?? "--")}</td>`).join("")}
          </tr>
        `).join("")}
      </tbody>
    </table>`;
}

function renderGridModelDeviceTree() {
  const container = $("modelDeviceTree");
  if (!container) return;
  const devices = gridModelDevices();
  const filter = state.modelDeviceFilter || { dev_type: "all", dev_name: "" };
  const groupEntries = groupedByDeviceType(devices).map(([devType, items]) => [
    devType,
    [...items].sort(compareModelRowsByIndex),
  ]);
  $("modelTreeSummary").textContent = `${groupEntries.length} 类 · ${devices.length} 台`;
  container.innerHTML = `
    <button
      type="button"
      class="tree-node tree-root ${filter.dev_type === "all" ? "is-active" : ""}"
      data-model-tree-type="all"
      data-model-tree-name=""
    >
      <span>全部设备</span>
      <strong>${devices.length}</strong>
    </button>
    ${groupEntries.map(([devType, items]) => {
      const isCollapsed = isDeviceTreeGroupCollapsed("model", devType);
      return `
      <div class="tree-group">
        <button
          type="button"
          class="tree-node tree-type ${isCollapsed ? "is-collapsed" : ""} ${filter.dev_type === devType && !filter.dev_name ? "is-active" : filter.dev_type === devType ? "is-parent-active" : ""}"
          data-model-tree-type="${escapeHtml(devType)}"
          data-model-tree-name=""
          ${deviceTreeTypeAttrs("model", devType, isCollapsed)}
        >
          ${deviceTreeTypeLabel(devType)}
          <strong>${items.length}</strong>
        </button>
        ${deviceTreeChildren(isCollapsed, items.map((dev) => {
            const idx = formatModelParamValue(dev.idx ?? dev.raw?.idx);
            return `
            <button
              type="button"
              class="tree-node tree-child model-tree-child ${filter.dev_type === dev.dev_type && filter.dev_name === dev.dev_name ? "is-active" : ""}"
              data-model-tree-type="${escapeHtml(dev.dev_type)}"
              data-model-tree-name="${escapeHtml(dev.dev_name)}"
            >
              <span class="model-tree-idx">${escapeHtml(idx)}</span>
              <span class="model-tree-name">${escapeHtml(dev.dev_name)}</span>
            </button>
          `;
          }).join(""))}
      </div>
    `;
    }).join("")}
  `;
}

function renderGridModelParamTable() {
  const container = $("modelParamTable");
  if (!container) return;
  const devices = gridModelDevices();
  const rows = filteredGridModelDevices(devices).map(modelAttributeRecordForDevice);
  const groups = groupedModelAttributeRecords(rows);
  const singleGroupColumnCount = groups.length === 1 ? modelAttributeColumns(groups[0][1]).length : 0;
  $("modelParamSummary").textContent = groups.length > 1
    ? `${gridModelFilterLabel()} · ${rows.length}/${devices.length} 台 · ${groups.length} 类表格`
    : `${gridModelFilterLabel()} · ${rows.length}/${devices.length} 台 · ${singleGroupColumnCount} 列属性`;
  if (!devices.length) {
    container.innerHTML = '<div class="empty-state">暂无电网模型数据</div>';
    return;
  }
  if (!rows.length) {
    container.innerHTML = '<div class="empty-state">当前筛选无模型参数</div>';
    return;
  }
  if (groups.length <= 1) {
    container.innerHTML = renderModelAttributeTable(rows);
    return;
  }
  container.innerHTML = groups.map(([devType, groupRows]) => {
    const columnCount = modelAttributeColumns(groupRows).length;
    return `
      <section class="model-param-group">
        <div class="model-param-group-head">
          <h3>${escapeHtml(devType)}</h3>
          <span>${groupRows.length} 台 · ${columnCount} 列属性</span>
        </div>
        ${renderModelAttributeTable(groupRows)}
      </section>
    `;
  }).join("");
}

function renderGridModelPage() {
  renderGridModelDeviceTree();
  renderGridModelParamTable();
}

function setGridModelFilter(devType, devName = "") {
  state.modelDeviceFilter = { dev_type: devType || "all", dev_name: devName || "" };
  renderGridModelPage();
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
    ${groupEntries.map(([devType, items]) => {
      const isCollapsed = isDeviceTreeGroupCollapsed("runtime", devType);
      return `
      <div class="tree-group">
        <button
          type="button"
          class="tree-node tree-type ${isCollapsed ? "is-collapsed" : ""} ${filter.dev_type === devType && !filter.dev_name ? "is-active" : filter.dev_type === devType ? "is-parent-active" : ""}"
          data-runtime-tree-type="${escapeHtml(devType)}"
          data-runtime-tree-name=""
          ${deviceTreeTypeAttrs("runtime", devType, isCollapsed)}
        >
          ${deviceTreeTypeLabel(devType)}
          <strong>${items.length}</strong>
        </button>
        ${deviceTreeChildren(isCollapsed, items.map((dev) => `
            <button
              type="button"
              class="tree-node tree-child ${filter.dev_type === dev.dev_type && filter.dev_name === dev.dev_name ? "is-active" : ""}"
              data-runtime-tree-type="${escapeHtml(dev.dev_type)}"
              data-runtime-tree-name="${escapeHtml(dev.dev_name)}"
            >
              <span>${escapeHtml(dev.dev_name)}</span>
              <small>${escapeHtml(deviceTreeBadge(dev))}</small>
            </button>
          `).join(""))}
      </div>
    `;
    }).join("")}
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

function traceAxisStepMinutes(windowMinutes) {
  const minutes = Math.max(1, Number(windowMinutes) || 60);
  if (minutes <= 15) return 5;
  if (minutes <= 60) return 15;
  if (minutes <= 180) return 30;
  if (minutes <= 360) return 60;
  if (minutes <= 1440) return 240;
  return Math.max(60, Math.round(minutes / 6 / 60) * 60);
}

function traceWindowAlignmentMinutes(windowMinutes) {
  const minutes = Math.max(1, Number(windowMinutes) || 60);
  if (minutes <= 15) return 15;
  if (minutes <= 1440) return minutes;
  return 1440;
}

function alignedTraceWindowRange(history, windowMinutes, fallbackMinute) {
  const alignmentMinutes = traceWindowAlignmentMinutes(windowMinutes);
  const axisStepMinutes = traceAxisStepMinutes(windowMinutes);
  const latestMinute = history.length ? history[history.length - 1].minute : fallbackMinute;
  const startMinute = Math.floor(latestMinute / alignmentMinutes) * alignmentMinutes;
  return {
    startMinute,
    endMinute: startMinute + windowMinutes,
    latestMinute,
    windowMinutes,
    alignmentMinutes,
    axisStepMinutes,
  };
}

function runtimeTraceWindowRange() {
  const history = state.runtimeTraceHistory || [];
  const windowMinutes = Math.max(1, Number(state.runtimeTraceWindowMinutes) || 60);
  const fallbackMinute = Number(state.snapshot?.clock?.absolute_minute ?? state.snapshot?.clock?.minute ?? 0) || 0;
  return alignedTraceWindowRange(history, windowMinutes, fallbackMinute);
}

function runtimeFormatClockMinute(minute) {
  const total = ((Math.round(minute) % 1440) + 1440) % 1440;
  const hours = Math.floor(total / 60);
  const minutes = total % 60;
  return `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:00`;
}

function runtimeAxisTickLabel(minute, range, index, lastIndex) {
  if (index === lastIndex) return runtimeFormatClockMinute(range.endMinute);
  return runtimeFormatClockMinute(minute);
}

function runtimeTraceAxisTicks(range, canvasWidth) {
  const maxTicks = canvasWidth < 480 ? 4 : canvasWidth < 760 ? 5 : 8;
  let step = range.axisStepMinutes || traceAxisStepMinutes(range.windowMinutes);
  while (Math.floor(range.windowMinutes / step) + 1 > maxTicks) {
    step *= 2;
  }
  const ticks = [];
  for (let minute = range.startMinute; minute <= range.endMinute + 1e-9; minute += step) {
    ticks.push(minute);
  }
  if (ticks[ticks.length - 1] !== range.endMinute) ticks.push(range.endMinute);
  return ticks;
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
  return alignedTraceWindowRange(history, windowMinutes, fallbackMinute);
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
    ${groupEntries.map(([devType, items]) => {
      const isCollapsed = isDeviceTreeGroupCollapsed("measurement", devType);
      return `
      <div class="tree-group">
        <button
          type="button"
          class="tree-node tree-type ${isCollapsed ? "is-collapsed" : ""} ${filter.dev_type === devType && !filter.dev_name ? "is-active" : filter.dev_type === devType ? "is-parent-active" : ""}"
          data-measurement-tree-type="${escapeHtml(devType)}"
          data-measurement-tree-name=""
          ${deviceTreeTypeAttrs("measurement", devType, isCollapsed)}
        >
          ${deviceTreeTypeLabel(devType)}
          <strong>${items.length}</strong>
        </button>
        ${deviceTreeChildren(isCollapsed, items.map((item) => `
            <button
              type="button"
              class="tree-node tree-child ${filter.dev_type === item.dev_type && filter.dev_name === item.dev_name ? "is-active" : ""}"
              data-measurement-tree-type="${escapeHtml(item.dev_type)}"
              data-measurement-tree-name="${escapeHtml(item.dev_name)}"
            >
              <span>${escapeHtml(item.dev_name)}</span>
              <small>${escapeHtml(item.count)}点</small>
            </button>
          `).join(""))}
      </div>
    `;
    }).join("")}
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
    ${groupEntries.map(([devType, items]) => {
      const isCollapsed = isDeviceTreeGroupCollapsed("faultDevice", devType);
      return `
      <div class="tree-group">
        <button
          type="button"
          class="tree-node tree-type ${isCollapsed ? "is-collapsed" : ""} ${filter.dev_type === devType && !filter.dev_name ? "is-active" : filter.dev_type === devType ? "is-parent-active" : ""}"
          data-fault-device-tree-type="${escapeHtml(devType)}"
          data-fault-device-tree-name=""
          ${deviceTreeTypeAttrs("faultDevice", devType, isCollapsed)}
        >
          ${deviceTreeTypeLabel(devType)}
          <strong>${items.length}</strong>
        </button>
        ${deviceTreeChildren(isCollapsed, items.map((dev) => `
            <button
              type="button"
              class="tree-node tree-child ${filter.dev_type === dev.dev_type && filter.dev_name === dev.dev_name ? "is-active" : ""}"
              data-fault-device-tree-type="${escapeHtml(dev.dev_type)}"
              data-fault-device-tree-name="${escapeHtml(dev.dev_name)}"
            >
              <span>${escapeHtml(dev.dev_name)}</span>
              <small>${escapeHtml(deviceTreeBadge(dev))}</small>
            </button>
          `).join(""))}
      </div>
    `;
    }).join("")}
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
    ${groupEntries.map(([devType, items]) => {
      const isCollapsed = isDeviceTreeGroupCollapsed("faultMeasurement", devType);
      return `
      <div class="tree-group">
        <button
          type="button"
          class="tree-node tree-type ${isCollapsed ? "is-collapsed" : ""} ${filter.dev_type === devType && !filter.dev_name ? "is-active" : filter.dev_type === devType ? "is-parent-active" : ""}"
          data-fault-measurement-tree-type="${escapeHtml(devType)}"
          data-fault-measurement-tree-name=""
          ${deviceTreeTypeAttrs("faultMeasurement", devType, isCollapsed)}
        >
          ${deviceTreeTypeLabel(devType)}
          <strong>${items.length}</strong>
        </button>
        ${deviceTreeChildren(isCollapsed, items.map((dev) => `
            <button
              type="button"
              class="tree-node tree-child ${filter.dev_type === dev.dev_type && filter.dev_name === dev.dev_name ? "is-active" : ""}"
              data-fault-measurement-tree-type="${escapeHtml(dev.dev_type)}"
              data-fault-measurement-tree-name="${escapeHtml(dev.dev_name)}"
            >
              <span>${escapeHtml(dev.dev_name)}</span>
              <small>${escapeHtml(dev.count)}点</small>
            </button>
          `).join(""))}
      </div>
    `;
    }).join("")}
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
    ${groupEntries.map(([devType, items]) => {
      const isCollapsed = isDeviceTreeGroupCollapsed("mode", devType);
      return `
      <div class="tree-group">
        <button
          type="button"
          class="tree-node tree-type ${isCollapsed ? "is-collapsed" : ""} ${filter.dev_type === devType && !filter.dev_name ? "is-active" : filter.dev_type === devType ? "is-parent-active" : ""}"
          data-mode-tree-type="${escapeHtml(devType)}"
          data-mode-tree-name=""
          ${deviceTreeTypeAttrs("mode", devType, isCollapsed)}
        >
          ${deviceTreeTypeLabel(devType)}
          <strong>${items.length}</strong>
        </button>
        ${deviceTreeChildren(isCollapsed, items.map((item) => `
            <button
              type="button"
              class="tree-node tree-child ${filter.dev_type === item.dev_type && filter.dev_name === item.dev_name ? "is-active" : ""}"
              data-mode-tree-type="${escapeHtml(item.dev_type)}"
              data-mode-tree-name="${escapeHtml(item.dev_name)}"
            >
              <span>${escapeHtml(item.dev_name)}</span>
              <small>${escapeHtml(item.mode)}</small>
            </button>
          `).join(""))}
      </div>
    `;
    }).join("")}
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
  button.addEventListener("click", () => controlClock(button.dataset.clock));
});
$("cloneModelButton").addEventListener("click", openCloneModelDialog);
$("closeCloneModelDialog").addEventListener("click", closeCloneModelDialog);
$("cancelCloneModel").addEventListener("click", closeCloneModelDialog);
$("cloneModelDialog").addEventListener("click", (event) => {
  if (event.target.id === "cloneModelDialog") closeCloneModelDialog();
});
$("cloneModelForm").addEventListener("submit", (event) => {
  event.preventDefault();
  cloneCurrentModel();
});
$("cloneModelName").addEventListener("input", () => validateCloneModelName());
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !$("cloneModelDialog").hidden) {
    closeCloneModelDialog();
  }
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
    if (faultDeviceTreeButton.dataset.treeToggleScope) {
      toggleDeviceTreeGroup(
        faultDeviceTreeButton.dataset.treeToggleScope,
        faultDeviceTreeButton.dataset.treeToggleGroup,
      );
    }
    setDeviceFaultFilter(
      faultDeviceTreeButton.dataset.faultDeviceTreeType,
      faultDeviceTreeButton.dataset.faultDeviceTreeName || "",
    );
  }
  const faultMeasurementTreeButton = event.target.closest("[data-fault-measurement-tree-type]");
  if (faultMeasurementTreeButton) {
    if (faultMeasurementTreeButton.dataset.treeToggleScope) {
      toggleDeviceTreeGroup(
        faultMeasurementTreeButton.dataset.treeToggleScope,
        faultMeasurementTreeButton.dataset.treeToggleGroup,
      );
    }
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
    if (measurementTreeButton.dataset.treeToggleScope) {
      toggleDeviceTreeGroup(
        measurementTreeButton.dataset.treeToggleScope,
        measurementTreeButton.dataset.treeToggleGroup,
      );
    }
    setMeasurementCompareFilter(
      measurementTreeButton.dataset.measurementTreeType,
      measurementTreeButton.dataset.measurementTreeName || "",
    );
  }
  const modelTreeButton = event.target.closest("[data-model-tree-type]");
  if (modelTreeButton) {
    if (modelTreeButton.dataset.treeToggleScope) {
      toggleDeviceTreeGroup(
        modelTreeButton.dataset.treeToggleScope,
        modelTreeButton.dataset.treeToggleGroup,
      );
    }
    setGridModelFilter(
      modelTreeButton.dataset.modelTreeType,
      modelTreeButton.dataset.modelTreeName || "",
    );
  }
  const runtimeTreeButton = event.target.closest("[data-runtime-tree-type]");
  if (runtimeTreeButton) {
    if (runtimeTreeButton.dataset.treeToggleScope) {
      toggleDeviceTreeGroup(
        runtimeTreeButton.dataset.treeToggleScope,
        runtimeTreeButton.dataset.treeToggleGroup,
      );
    }
    setRuntimeDeviceFilter(
      runtimeTreeButton.dataset.runtimeTreeType,
      runtimeTreeButton.dataset.runtimeTreeName || "",
    );
  }
  const modeTreeButton = event.target.closest("[data-mode-tree-type]");
  if (modeTreeButton) {
    if (modeTreeButton.dataset.treeToggleScope) {
      toggleDeviceTreeGroup(
        modeTreeButton.dataset.treeToggleScope,
        modeTreeButton.dataset.treeToggleGroup,
      );
    }
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
