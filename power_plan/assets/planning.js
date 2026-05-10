const state = {
  schemes: [],
  currentScheme: "",
  payload: null,
  month: 0,
  timeSeriesLoading: null,
  mapConfig: null,
  mapPoint: null,
  mapInstance: null,
  mapMarker: null,
  chartMeta: null,
  timeChartManualHeight: null,
  layoutObserver: null,
};

const deviceSpecs = [
  ["diesel_generators", "柴发", ["name", "capacity", "cost", "power_upper", "power_lower", "fuel_rate", "quantity_lower", "quantity_upper"]],
  ["wind_turbines", "风机", ["name", "capacity", "cost", "cut_in_wind_speed", "cut_out_wind_speed", "quantity_lower", "quantity_upper"]],
  ["photovoltaics", "光伏", ["name", "capacity", "cost", "generation_efficiency", "quantity_lower", "quantity_upper"]],
  ["storage_pcs", "储能PCS", ["name", "power_capacity", "cost", "quantity_lower", "quantity_upper"]],
  ["storage_battery_packs", "储能电池组", ["name", "battery_capacity", "cost", "quantity_lower", "quantity_upper"]],
  ["hydrogen_electrolyzers", "电制氢", ["name", "power_capacity", "cost", "electric_to_hydrogen_efficiency", "quantity_lower", "quantity_upper"]],
  ["hydrogen_tanks", "储氢罐", ["name", "hydrogen_tank_capacity", "cost", "quantity_lower", "quantity_upper"]],
  ["fuel_cells", "燃料电池", ["name", "power_capacity", "cost", "hydrogen_to_electric_efficiency", "quantity_lower", "quantity_upper"]],
];

const summarySeries = [
  ["wind_speed", "风速", "#1f9bb4", "m/s"],
  ["solar_irradiance", "太阳辐照", "#d79018", "W/m2"],
  ["temperature", "温度", "#7a5aa6", "℃"],
  ["load", "负荷", "#2d6b45", "kW"],
];

const planningParameterSpecs = [
  ["design_life_years", "设计使用年限(年)", "number", { min: 1, integer: true, defaultValue: 20 }],
  ["diesel_price", "柴油价格(万元/吨)", "number", { min: 0, defaultValue: 0 }],
  ["planning_load_factor", "规划负荷系数(0.1-10.0)", "number", { min: 0.1, max: 10, defaultValue: 1 }],
  ["green_power_ratio_lower", "绿电电量占比下限(0.0-1.0)", "number", { min: 0, max: 1, defaultValue: 0 }],
  ["storage_frequency_regulation_enabled", "储能是否参与调频", "boolean", { defaultValue: false }],
  ["load_disturbance_factor", "负荷扰动系数(0.0-0.5)", "number", { min: 0, max: 0.5, defaultValue: 0 }],
  ["frequency_security_constraint_enabled", "是否考虑频率安全约束", "boolean", { defaultValue: false }],
  ["frequency_security_upper", "频率安全上限(1.0-1.5)", "number", { min: 1, max: 1.5, defaultValue: 1.5 }],
  ["frequency_security_lower", "频率安全下限(1.0-1.5)", "number", { min: 1, max: 1.5, defaultValue: 1.0 }],
  ["post_disturbance_power_balance_enabled", "是否考虑扰动后功率平衡", "boolean", { defaultValue: false }],
  ["renewable_n_1_enabled", "是否考虑新能源N-1", "boolean", { defaultValue: false }],
  ["load_disturbance_enabled", "是否考虑负荷扰动", "boolean", { defaultValue: false }],
];

const visibleDevices = new Set(deviceSpecs.map(([key]) => key));

const monthRanges = [
  ["1月", 0, 744],
  ["2月", 744, 1416],
  ["3月", 1416, 2160],
  ["4月", 2160, 2880],
  ["5月", 2880, 3624],
  ["6月", 3624, 4344],
  ["7月", 4344, 5088],
  ["8月", 5088, 5832],
  ["9月", 5832, 6552],
  ["10月", 6552, 7296],
  ["11月", 7296, 8016],
  ["12月", 8016, 8760],
];

const deviceGroups = [
  ["windSolarDiesel", "风光柴", ["diesel_generators", "wind_turbines", "photovoltaics"]],
  ["electricStorage", "电储能", ["storage_pcs", "storage_battery_packs"]],
  ["hydrogenStorage", "氢储能", ["hydrogen_electrolyzers", "hydrogen_tanks", "fuel_cells"]],
];

const labels = {
  name: "名称",
  solar_irradiance: "太阳辐照",
  temperature: "温度",
  capacity: "容量",
  power_capacity: "功率容量",
  battery_capacity: "电池容量",
  hydrogen_tank_capacity: "储氢罐容量",
  quantity_lower: "数据下限(台)",
  quantity_upper: "数据上限(台)",
  cost: "成本(万元/台)",
  power_upper: "功率上限(kW)",
  power_lower: "功率下限(kW)",
  fuel_rate: "油耗率(kg/kWh)",
  cut_in_wind_speed: "切入风速(m/s)",
  cut_out_wind_speed: "切出风速(m/s)",
  generation_efficiency: "发电效率(0-1.0)",
  electric_to_hydrogen_efficiency: "电-氢效率(Nm3/kWh)",
  hydrogen_to_electric_efficiency: "氢-电效率(kWh/Nm3)",
};

document.addEventListener("DOMContentLoaded", () => {
  bindTabs();
  bindSummaryTabs();
  bindTimeResizeHandle();
  bindActions();
  bindAdaptiveLayout();
  syncAdaptiveLayout();
  loadSchemes().catch(showError);
});

function bindTabs() {
  document.querySelectorAll(".tab").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((item) => item.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((item) => item.classList.remove("active"));
      button.classList.add("active");
      document.getElementById(`${button.dataset.tab}Tab`).classList.add("active");
      syncAdaptiveLayout();
      ensureTimeSeriesForActiveTab();
    });
  });
}

function bindSummaryTabs() {
  const buttons = Array.from(document.querySelectorAll("[data-summary-tab]"));
  const panels = Array.from(document.querySelectorAll("[data-summary-panel]"));
  buttons.forEach((button) => {
    button.addEventListener("click", () => {
      const target = button.dataset.summaryTab;
      buttons.forEach((item) => {
        const active = item === button;
        item.classList.toggle("active", active);
        item.setAttribute("aria-selected", String(active));
      });
      panels.forEach((panel) => {
        const active = panel.dataset.summaryPanel === target;
        panel.classList.toggle("active", active);
        panel.hidden = !active;
      });
      syncAdaptiveLayout();
    });
  });
}

function bindActions() {
  document.getElementById("createScheme").addEventListener("click", createScheme);
  document.getElementById("copyScheme").addEventListener("click", copyScheme);
  document.getElementById("renameScheme").addEventListener("click", renameScheme);
  document.getElementById("saveScheme").addEventListener("click", saveScheme);
  document.getElementById("deleteScheme").addEventListener("click", deleteScheme);
  document.getElementById("geocodePlace").addEventListener("click", geocodePlace);
  document.getElementById("fetchWeatherHistory").addEventListener("click", fetchWeatherHistory);
  document.getElementById("openCoordinatePicker").addEventListener("click", openCoordinatePicker);
  document.getElementById("closeMapPicker").addEventListener("click", closeMapPicker);
  document.getElementById("confirmMapPoint").addEventListener("click", confirmMapPoint);
  document.querySelectorAll("[data-curve]").forEach((button) => {
    button.addEventListener("click", () => selectCurve(button.dataset.curve));
  });
  document.getElementById("timeChart").addEventListener("mousemove", onChartMouseMove);
  document.getElementById("timeChart").addEventListener("mouseleave", hideChartCursor);
  document.addEventListener("mousemove", onHistogramMouseMove);
  document.addEventListener("mouseleave", hideHistogramTip);
  window.addEventListener("resize", syncAdaptiveLayout);
}

function bindAdaptiveLayout() {
  if (!("ResizeObserver" in window)) return;
  const targets = [
    document.querySelector(".editor-panel"),
    document.getElementById("timeTab"),
    document.getElementById("devicesTab"),
    document.getElementById("planningTab"),
    document.getElementById("limitsTab"),
  ].filter(Boolean);
  state.layoutObserver = new ResizeObserver(() => syncAdaptiveLayout());
  targets.forEach((target) => state.layoutObserver.observe(target));
}

function syncAdaptiveLayout() {
  applyPanelTableMaxHeight();
  applyAdaptiveTimeSeriesLayout();
  applyAdaptiveSummaryLayout();
  renderChart();
}

function applyPanelTableMaxHeight() {
  const editor = document.querySelector(".editor-panel");
  const header = document.querySelector(".editor-header");
  const available = editor ? editor.clientHeight - (header?.offsetHeight || 0) - 72 : window.innerHeight * 0.55;
  const tableMaxHeight = Math.min(680, Math.max(220, available));
  document.documentElement.style.setProperty("--panel-table-max-height", `${Math.round(tableMaxHeight)}px`);
}

function applyAdaptiveTimeSeriesLayout() {
  const tab = document.getElementById("timeTab");
  const chart = document.getElementById("timeChart");
  const chartCard = tab?.querySelector(".chart-card");
  const tableCard = tab?.querySelector(".table-card");
  const table = document.getElementById("timeTable");
  const toolbar = tab?.querySelector(".time-table-toolbar");
  const handle = document.getElementById("timeResizeHandle");
  if (!tab || !chart || !chartCard || !tableCard || !table || !tab.classList.contains("active")) return;

  const tabHeight = tab.clientHeight || Math.max(520, window.innerHeight - 180);
  const chartChrome = Math.max(0, chartCard.offsetHeight - chart.clientHeight);
  const tableChrome = Math.max(0, tableCard.offsetHeight - table.clientHeight) || (toolbar?.offsetHeight || 0) + 44;
  const handleHeight = (handle?.offsetHeight || 14) + 10;
  const available = Math.max(240, tabHeight - chartChrome - tableChrome - handleHeight - 32);
  const autoChartHeight = Math.min(340, Math.max(140, available * 0.4));
  const chartHeight = clampTimeChartHeight(state.timeChartManualHeight ?? autoChartHeight);
  const tableHeight = Math.min(620, Math.max(120, available - chartHeight));

  document.documentElement.style.setProperty("--time-chart-height", `${Math.round(chartHeight)}px`);
  document.documentElement.style.setProperty("--time-table-height", `${Math.round(tableHeight)}px`);
  handle?.setAttribute("aria-valuenow", String(Math.round(chartHeight)));
}

function applyAdaptiveSummaryLayout() {
  const tab = document.getElementById("limitsTab");
  const summarySwitcher = tab?.querySelector(".summary-switcher");
  const summaryTabs = tab?.querySelector(".summary-tabs");
  const activePanel = tab?.querySelector(".summary-tab-panel.active");
  const heading = activePanel?.querySelector(".panel-heading");
  if (!tab || !summarySwitcher || !activePanel || !tab.classList.contains("active")) return;

  const switcherStyle = getComputedStyle(summarySwitcher);
  const tabsStyle = summaryTabs ? getComputedStyle(summaryTabs) : null;
  const headingStyle = heading ? getComputedStyle(heading) : null;
  const switcherPadding = parseFloat(switcherStyle.paddingTop || 0) + parseFloat(switcherStyle.paddingBottom || 0);
  const tabsHeight = (summaryTabs?.offsetHeight || 0) + parseFloat(tabsStyle?.marginBottom || 0);
  const headingHeight = (heading?.offsetHeight || 0) + parseFloat(headingStyle?.marginBottom || 0);
  const availablePanelHeight = (summarySwitcher.clientHeight || Math.max(260, tab.clientHeight - 32)) - switcherPadding - tabsHeight;
  const panelHeight = Math.max(180, availablePanelHeight);
  const contentHeight = Math.max(140, panelHeight - headingHeight);
  const tableHeight = Math.max(160, contentHeight);
  const histogramGridHeight = Math.max(240, contentHeight);
  const histogramSvgHeight = Math.max(120, (histogramGridHeight - 14) / 2 - 76);

  document.documentElement.style.setProperty("--summary-panel-height", `${Math.round(panelHeight)}px`);
  document.documentElement.style.setProperty("--summary-table-height", `${Math.round(tableHeight)}px`);
  document.documentElement.style.setProperty("--summary-histogram-grid-height", `${Math.round(histogramGridHeight)}px`);
  document.documentElement.style.setProperty("--summary-histogram-svg-height", `${Math.round(histogramSvgHeight)}px`);
}

function bindTimeResizeHandle() {
  const handle = document.getElementById("timeResizeHandle");
  const chart = document.getElementById("timeChart");
  if (!handle || !chart) return;

  const applyHeight = (height) => {
    const safeHeight = clampTimeChartHeight(height);
    state.timeChartManualHeight = safeHeight;
    document.documentElement.style.setProperty("--time-chart-height", `${Math.round(safeHeight)}px`);
    handle.setAttribute("aria-valuenow", String(Math.round(safeHeight)));
    syncAdaptiveLayout();
  };

  handle.addEventListener("pointerdown", (event) => {
    event.preventDefault();
    const startY = event.clientY;
    const startHeight = chart.getBoundingClientRect().height || 240;
    handle.classList.add("dragging");
    handle.setPointerCapture?.(event.pointerId);

    const onMove = (moveEvent) => {
      applyHeight(startHeight + moveEvent.clientY - startY);
    };
    const onDone = () => {
      handle.classList.remove("dragging");
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onDone);
      window.removeEventListener("pointercancel", onDone);
      syncAdaptiveLayout();
    };

    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onDone);
    window.addEventListener("pointercancel", onDone);
  });

  handle.addEventListener("keydown", (event) => {
    const currentHeight = chart.getBoundingClientRect().height || 240;
    const keySteps = {
      ArrowUp: -12,
      ArrowDown: 12,
      PageUp: -48,
      PageDown: 48,
    };
    if (event.key in keySteps) {
      event.preventDefault();
      applyHeight(currentHeight + keySteps[event.key]);
    } else if (event.key === "Home") {
      event.preventDefault();
      applyHeight(120);
    } else if (event.key === "End") {
      event.preventDefault();
      applyHeight(maxTimeChartHeight());
    }
  });

  handle.setAttribute("aria-valuenow", String(Math.round(chart.getBoundingClientRect().height || 240)));
}

function clampTimeChartHeight(height) {
  return Math.min(Math.max(Number(height) || 240, 120), maxTimeChartHeight());
}

function maxTimeChartHeight() {
  const tab = document.getElementById("timeTab");
  const available = tab ? tab.clientHeight - 230 : 420;
  return Math.max(140, Math.min(520, available));
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.message || data.error || "请求失败");
  return data;
}

async function loadSchemes() {
  state.schemes = (await api("/api/planning/schemes")).schemes;
  renderSchemes();
  if (!state.currentScheme && state.schemes.length) {
    await selectScheme(state.schemes[0].name);
  } else {
    renderSummary();
  }
}

function renderSchemes() {
  const list = document.getElementById("schemeList");
  if (!state.schemes.length) {
    list.innerHTML = "<div class=\"validation-item\">暂无方案，请新建方案。</div>";
    return;
  }
  list.innerHTML = state.schemes
    .map((scheme) => `<button class="scheme-item ${scheme.name === state.currentScheme ? "active" : ""}" type="button" data-name="${escapeHtml(scheme.name)}">${escapeHtml(scheme.name)}</button>`)
    .join("");
  document.querySelectorAll(".scheme-item").forEach((button) => {
    button.addEventListener("click", () => selectScheme(button.dataset.name).catch(showError));
  });
}

async function selectScheme(name) {
  state.currentScheme = name;
  state.timeSeriesLoading = null;
  state.payload = normalizePayload(await api(`/api/planning/schemes/${encodeURIComponent(name)}/overview`));
  state.month = 0;
  renderAll();
  ensureTimeSeriesForActiveTab();
}

async function createScheme() {
  const name = normalizeSchemeName(prompt("请输入新方案名称"));
  if (!name) return;
  if (schemeNameExists(name)) {
    alert("方案名称已存在，请使用其他名称");
    return;
  }
  const created = await api("/api/planning/schemes", { method: "POST", body: JSON.stringify({ name }) }).catch(showError);
  if (!created) return;
  state.currentScheme = created.scheme;
  await loadSchemes();
  await selectScheme(state.currentScheme);
}

async function copyScheme() {
  if (!state.currentScheme) return alert("请先选择方案");
  const target = normalizeSchemeName(prompt("请输入复制后的方案名称", `${state.currentScheme}_副本`));
  if (!target) return;
  if (schemeNameExists(target)) {
    alert("方案名称已存在，请使用其他名称");
    return;
  }
  const copied = await api("/api/planning/schemes/copy", {
    method: "POST",
    body: JSON.stringify({ source: state.currentScheme, target }),
  }).catch(showError);
  if (!copied) return;
  state.currentScheme = copied.scheme;
  await loadSchemes();
  await selectScheme(state.currentScheme);
}

async function renameScheme() {
  if (!state.currentScheme) return alert("请先选择方案");
  const target = normalizeSchemeName(prompt("请输入新的方案名称", state.currentScheme));
  if (!target || target === state.currentScheme) return;
  if (schemeNameExists(target, state.currentScheme)) {
    alert("方案名称已存在，请使用其他名称");
    return;
  }
  const renamed = await api("/api/planning/schemes/rename", {
    method: "POST",
    body: JSON.stringify({ source: state.currentScheme, target }),
  }).catch(showError);
  if (!renamed) return;
  state.currentScheme = renamed.scheme;
  await loadSchemes();
  await selectScheme(state.currentScheme);
}

async function saveScheme() {
  if (!state.currentScheme || !state.payload) return alert("请先选择方案");
  if (!isTimeSeriesLoaded()) {
    await ensureTimeSeriesLoaded().catch(showError);
    if (!isTimeSeriesLoaded()) return;
  }
  const warnings = collectSaveWarnings();
  if (warnings.length) {
    renderSummary();
    alert(`参数校验未通过：\n${warnings.map((item) => `- ${item.message}`).join("\n")}`);
    return;
  }
  state.payload = normalizePayload(await api(`/api/planning/schemes/${encodeURIComponent(state.currentScheme)}`, {
    method: "PUT",
    body: JSON.stringify(state.payload),
  }).catch(showError));
  if (!state.payload) return;
  renderAll();
  alert("保存成功");
}

async function deleteScheme() {
  if (!state.currentScheme) return alert("请先选择方案");
  const deletedIndex = state.schemes.findIndex((scheme) => scheme.name === state.currentScheme);
  if (!confirm(`确认删除方案“${state.currentScheme}”？删除后无法恢复。`)) return;
  const result = await api(`/api/planning/schemes/${encodeURIComponent(state.currentScheme)}`, { method: "DELETE" }).catch(showError);
  if (!result) return;
  await selectNextSchemeAfterDelete(Math.max(0, deletedIndex));
  alert("删除成功");
}

async function geocodePlace() {
  const placeInput = document.getElementById("weatherPlace");
  const place = placeInput.value.trim();
  if (!place) {
    setMapPickerHint("请输入地名");
    return;
  }
  setMapPickerHint("正在获取坐标...");
  const result = await api("/api/planning/geocode", {
    method: "POST",
    body: JSON.stringify({ place }),
  }).catch((error) => {
    setMapPickerHint(error.message || String(error));
    return null;
  });
  if (!result) return;
  setMapPoint(result.latitude, result.longitude, "geocode");
}

async function fetchWeatherHistory() {
  if (!state.currentScheme || !state.payload) {
    setWeatherImportStatus("请先选择方案", "error");
    return;
  }
  const latitude = Number(document.getElementById("weatherLatitude").value);
  const longitude = Number(document.getElementById("weatherLongitude").value);
  const year = Number(document.getElementById("weatherYear").value);
  const localError = validateWeatherInputs(latitude, longitude, year);
  if (localError) {
    setWeatherImportStatus(localError, "error");
    return;
  }
  setWeatherImportStatus("正在获取历史气象...");
  const result = await api("/api/planning/weather-history", {
    method: "POST",
    body: JSON.stringify({ latitude, longitude, year }),
  }).catch((error) => {
    setWeatherImportStatus(error.message || String(error), "error");
    return null;
  });
  if (!result) return;
  const rows = Array.isArray(result.rows) ? result.rows : [];
  if (rows.length !== 8760) {
    setWeatherImportStatus(`历史气象数据小时数应为8760，当前为${rows.length}`, "error");
    return;
  }
  await ensureTimeSeriesLoaded().catch((error) => {
    setWeatherImportStatus(error.message || String(error), "error");
    return false;
  });
  if (!isTimeSeriesLoaded()) return;
  const nextRows = (state.payload.time_series || []).map((row, index) => {
    const weather = rows[index];
    if (!weather) return row;
    return {
      ...row,
      datetime: weather.datetime || row.datetime,
      wind_speed: weather.wind_speed,
      solar_irradiance: weather.solar_irradiance,
      temperature: weather.temperature,
    };
  });
  if (nextRows.length !== 8760) {
    setWeatherImportStatus("当前时序表不是8760行，未更新数据", "error");
    return;
  }
  state.payload.time_series = nextRows;
  state.payload.time_series_count = nextRows.length;
  setTimeSeriesLoaded(true);
  renderChart();
  renderTimeTable();
  renderLimitSummary();
  renderSummary();
  setWeatherImportStatus(`${year}年气象已更新`, "ok");
}

async function openCoordinatePicker() {
  const modal = document.getElementById("mapPickerModal");
  modal.hidden = false;
  setMapPickerHint("根据地名查找坐标，或点击地图选点。");
  const config = await loadMapConfig();
  if (!config || !config.amap_key) {
    setMapPickerHint("可按地名查找；未配置地图 Key。");
    return;
  }
  try {
    await loadAmapScript(config.amap_key);
    initAmapPicker();
    setMapPickerHint("根据地名查找坐标，或点击地图选点。");
  } catch (error) {
    setMapPickerHint(`地图加载失败：${error.message || error}`);
  }
}

function closeMapPicker() {
  document.getElementById("mapPickerModal").hidden = true;
}

async function loadMapConfig() {
  if (state.mapConfig) return state.mapConfig;
  state.mapConfig = await api("/api/planning/map-config").catch((error) => {
    document.getElementById("mapPickerHint").textContent = error.message || String(error);
    return null;
  });
  return state.mapConfig;
}

function loadAmapScript(key) {
  if (window.AMap) return Promise.resolve();
  if (window.__powerPlanAmapLoading) return window.__powerPlanAmapLoading;
  window.__powerPlanAmapLoading = new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = `https://webapi.amap.com/maps?v=2.0&key=${encodeURIComponent(key)}`;
    script.async = true;
    script.onload = () => resolve();
    script.onerror = () => reject(new Error("高德地图脚本加载失败"));
    document.head.appendChild(script);
  });
  return window.__powerPlanAmapLoading;
}

function initAmapPicker() {
  const latitude = Number(document.getElementById("weatherLatitude").value);
  const longitude = Number(document.getElementById("weatherLongitude").value);
  const center = Number.isFinite(latitude) && Number.isFinite(longitude) ? [longitude, latitude] : [116.39723, 39.9075];
  if (!state.mapInstance) {
    state.mapInstance = new window.AMap.Map("mapPickerCanvas", {
      zoom: 5,
      center,
      resizeEnable: true,
    });
    state.mapMarker = new window.AMap.Marker({ position: center });
    state.mapInstance.add(state.mapMarker);
    state.mapInstance.on("click", (event) => {
      const point = event.lnglat;
      setMapPoint(point.getLat(), point.getLng());
    });
  } else {
    state.mapInstance.setCenter(center);
  }
  setTimeout(() => state.mapInstance.resize?.(), 80);
}

function setMapPoint(latitude, longitude, source = "map") {
  state.mapPoint = { latitude, longitude };
  document.getElementById("weatherLatitude").value = Number(latitude).toFixed(6);
  document.getElementById("weatherLongitude").value = Number(longitude).toFixed(6);
  if (state.mapInstance) {
    state.mapInstance.setCenter([longitude, latitude]);
  }
  if (state.mapMarker) {
    state.mapMarker.setPosition([longitude, latitude]);
  }
  const sourceText = source === "geocode" ? "地名坐标" : "地图坐标";
  setMapPickerHint(`${sourceText}：${Number(latitude).toFixed(6)}, ${Number(longitude).toFixed(6)}`);
  setWeatherImportStatus("坐标已填入", "ok");
}

function confirmMapPoint() {
  if (!state.mapPoint) {
    document.getElementById("mapPickerHint").textContent = "请先点击地图选择位置";
    return;
  }
  closeMapPicker();
}

function validateWeatherInputs(latitude, longitude, year) {
  const currentYear = new Date().getFullYear();
  if (!Number.isFinite(latitude) || latitude < -90 || latitude > 90) return "纬度范围应为 -90 到 90";
  if (!Number.isFinite(longitude) || longitude < -180 || longitude > 180) return "经度范围应为 -180 到 180";
  if (!Number.isInteger(year) || year < 2001 || year >= currentYear) return `历史数据年必须为2001到${currentYear - 1}之间的整数`;
  return "";
}

function setWeatherImportStatus(message, level = "") {
  const host = document.getElementById("weatherImportStatus");
  if (!host) return;
  host.textContent = message;
  host.classList.toggle("error", level === "error");
  host.classList.toggle("ok", level === "ok");
}

function setMapPickerHint(message) {
  const hint = document.getElementById("mapPickerHint");
  if (hint) hint.textContent = message;
}

async function selectNextSchemeAfterDelete(deletedIndex) {
  state.schemes = (await api("/api/planning/schemes")).schemes;
  const nextScheme = state.schemes[Math.min(deletedIndex, state.schemes.length - 1)];
  state.currentScheme = "";
  state.payload = null;
  if (nextScheme) {
    await selectScheme(nextScheme.name);
  } else {
    renderAll();
  }
}

async function ensureTimeSeriesLoaded() {
  if (!state.currentScheme || !state.payload || isTimeSeriesLoaded()) return true;
  if (state.timeSeriesLoading) return state.timeSeriesLoading;
  state.timeSeriesLoading = api(`/api/planning/schemes/${encodeURIComponent(state.currentScheme)}/time-series`)
    .then((data) => {
      if (!state.payload || data.scheme !== state.currentScheme) return false;
      state.payload.time_series = data.time_series || [];
      state.payload.time_series_count = data.time_series_count ?? state.payload.time_series.length;
      state.payload.validation = data.validation || state.payload.validation || [];
      setTimeSeriesLoaded(true);
      state.month = 0;
      renderChart();
      renderTimeTable();
      renderLimitSummary();
      renderSummary();
      return true;
    })
    .finally(() => {
      state.timeSeriesLoading = null;
    });
  renderChart();
  renderMonthTabs();
  renderTimeTable();
  renderLimitSummary();
  renderSummary();
  return state.timeSeriesLoading;
}

function ensureTimeSeriesForActiveTab() {
  if (shouldAutoLoadTimeSeries()) {
    ensureTimeSeriesLoaded().catch(showError);
  }
}

function shouldAutoLoadTimeSeries() {
  const tab = activeTabKey();
  return tab === "time" || tab === "limits";
}

function activeTabKey() {
  const tab = document.querySelector(".tab.active");
  return tab ? tab.dataset.tab : "time";
}

function renderAll() {
  renderSchemes();
  renderChart();
  renderMonthTabs();
  renderTimeTable();
  renderDeviceFilters();
  renderDeviceTables();
  renderPlanningParameters();
  renderLimitSummary();
  renderSummary();
}

function renderChart() {
  const svg = document.getElementById("timeChart");
  if (!state.payload) {
    svg.innerHTML = "";
    state.chartMeta = null;
    return;
  }
  if (!isTimeSeriesLoaded()) {
    const width = svg.clientWidth || 900;
    const height = svg.clientHeight || 320;
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    svg.innerHTML = `<rect x="0" y="0" width="${width}" height="${height}" rx="18" fill="transparent"/><text x="${width / 2}" y="${height / 2}" text-anchor="middle" fill="#5a716e" font-size="16">${state.timeSeriesLoading ? "时序数据加载中..." : "时序数据尚未加载"}</text>`;
    state.chartMeta = null;
    return;
  }
  const rows = state.payload.time_series || [];
  const width = svg.clientWidth || 900;
  const height = svg.clientHeight || 320;
  const padLeft = 62;
  const padRight = 28;
  const padTop = 28;
  const padBottom = 48;
  const plotWidth = Math.max(1, width - padLeft - padRight);
  const plotHeight = Math.max(1, height - padTop - padBottom);
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  const [curveKey, curveTitle, color, unit] = selectedCurveSpec();
  const values = numericValues(rows, curveKey);
  const rawMin = values.length ? Math.min(...values) : 0;
  const rawMax = values.length ? Math.max(...values) : 1;
  const minValue = rawMin === rawMax ? rawMin - 1 : rawMin;
  const maxValue = rawMin === rawMax ? rawMax + 1 : rawMax;
  const valueSpan = maxValue - minValue || 1;
  const x = (index) => padLeft + (index / Math.max(1, rows.length - 1)) * plotWidth;
  const y = (value) => {
    const number = Number(value);
    const safeValue = Number.isFinite(number) ? number : 0;
    return padTop + plotHeight - ((safeValue - minValue) / valueSpan) * plotHeight;
  };
  const yTicks = [0, 1, 2, 3, 4].map((index) => minValue + (valueSpan * index) / 4);
  const yGrid = yTicks
    .map((value) => {
      const tickY = y(value);
      return `<line x1="${padLeft}" x2="${width - padRight}" y1="${tickY.toFixed(1)}" y2="${tickY.toFixed(1)}" stroke="#d4e1dd"/><text x="${padLeft - 8}" y="${(tickY + 4).toFixed(1)}" text-anchor="end" fill="#5a716e" font-size="11">${escapeHtml(formatNumber(value))}</text>`;
    })
    .join("");
  const xTicks = monthRanges
    .map(([label, start]) => {
      const tickX = x(start);
      return `<line x1="${tickX.toFixed(1)}" x2="${tickX.toFixed(1)}" y1="${padTop + plotHeight}" y2="${padTop + plotHeight + 5}" stroke="#8ba49f"/><text x="${tickX.toFixed(1)}" y="${height - 12}" text-anchor="middle" fill="#5a716e" font-size="11">${label}</text>`;
    })
    .join("");
  const d = rows.map((row, index) => `${index === 0 ? "M" : "L"}${x(index).toFixed(1)},${y(row[curveKey]).toFixed(1)}`).join(" ");
  const axisTitle = `${curveTitle}${unit ? `(${unit})` : ""}`;
  svg.innerHTML = `<rect x="0" y="0" width="${width}" height="${height}" rx="18" fill="transparent"/><g>${yGrid}</g><line x1="${padLeft}" x2="${width - padRight}" y1="${padTop + plotHeight}" y2="${padTop + plotHeight}" stroke="#5a716e"/><line x1="${padLeft}" x2="${padLeft}" y1="${padTop}" y2="${padTop + plotHeight}" stroke="#5a716e"/><g>${xTicks}</g><path d="${d}" fill="none" stroke="${color}" stroke-width="2" vector-effect="non-scaling-stroke"/><g id="chartCursor" class="chart-cursor" hidden><line id="chartCursorLine" x1="0" x2="0" y1="${padTop}" y2="${padTop + plotHeight}"/><circle id="chartCursorPoint" cx="0" cy="0" r="4"/></g><text x="${padLeft}" y="18" fill="#294944" font-size="13" font-weight="700">${escapeHtml(axisTitle)}</text>`;
  state.chartMeta = { rows, curveKey, curveTitle, color, unit, padLeft, padRight, padTop, plotWidth, plotHeight, minValue, valueSpan, width, height };
}

function selectedCurveSpec() {
  const selected = document.querySelector('[data-curve][aria-pressed="true"]');
  return summarySeries.find(([key]) => key === selected?.dataset.curve) || summarySeries[0];
}

function selectCurve(curveKey) {
  const target = summarySeries.find(([key]) => key === curveKey) || summarySeries[0];
  document.querySelectorAll("[data-curve]").forEach((button) => {
    const active = button.dataset.curve === target[0];
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  });
  hideChartCursor();
  renderChart();
}

function onChartMouseMove(event) {
  if (!state.chartMeta) return;
  const svg = document.getElementById("timeChart");
  const cursor = document.getElementById("chartCursor");
  const cursorLine = document.getElementById("chartCursorLine");
  const cursorPoint = document.getElementById("chartCursorPoint");
  const tip = document.getElementById("chartTip");
  if (!svg || !cursor || !cursorLine || !cursorPoint || !tip) return;
  const meta = state.chartMeta;
  const rect = svg.getBoundingClientRect();
  const localX = ((event.clientX - rect.left) / Math.max(1, rect.width)) * meta.width;
  const ratio = Math.min(1, Math.max(0, (localX - meta.padLeft) / meta.plotWidth));
  const index = Math.round(ratio * Math.max(1, meta.rows.length - 1));
  const row = meta.rows[index];
  if (!row) return;
  const pointX = meta.padLeft + (index / Math.max(1, meta.rows.length - 1)) * meta.plotWidth;
  const value = Number(row[meta.curveKey]);
  const safeValue = Number.isFinite(value) ? value : 0;
  const pointY = meta.padTop + meta.plotHeight - ((safeValue - meta.minValue) / meta.valueSpan) * meta.plotHeight;
  cursor.hidden = false;
  cursorLine.setAttribute("x1", pointX.toFixed(1));
  cursorLine.setAttribute("x2", pointX.toFixed(1));
  cursorPoint.setAttribute("cx", pointX.toFixed(1));
  cursorPoint.setAttribute("cy", pointY.toFixed(1));
  cursorPoint.setAttribute("fill", meta.color);
  tip.hidden = false;
  tip.innerHTML = `${escapeHtml(meta.curveTitle)}：${escapeHtml(formatNumber(safeValue))}${escapeHtml(meta.unit)}<br>小时：${escapeHtml(row.hour_index ?? index + 1)}<br>时间：${escapeHtml(row.datetime || "")}`;
  tip.style.left = `${Math.min(window.innerWidth - 180, event.clientX + 14)}px`;
  tip.style.top = `${Math.max(12, event.clientY - 28)}px`;
}

function hideChartCursor() {
  const cursor = document.getElementById("chartCursor");
  const tip = document.getElementById("chartTip");
  if (cursor) cursor.hidden = true;
  if (tip) tip.hidden = true;
}

function renderMonthTabs() {
  const host = document.getElementById("monthTabs");
  if (!host) return;
  host.innerHTML = monthRanges
    .map(([label], index) => `<button class="month-tab ${index === state.month ? "active" : ""}" type="button" data-month="${index}">${label}</button>`)
    .join("");
  host.querySelectorAll("[data-month]").forEach((button) => {
    button.addEventListener("click", () => {
      state.month = Number(button.dataset.month);
      renderMonthTabs();
      renderTimeTable();
    });
  });
}

function renderTimeTable() {
  const container = document.getElementById("timeTable");
  if (!state.payload) {
    container.innerHTML = "";
    return;
  }
  if (!isTimeSeriesLoaded()) {
    document.getElementById("pageInfo").textContent = state.timeSeriesLoading ? "加载中" : "未加载";
    container.innerHTML = `<div class="empty-summary">${state.timeSeriesLoading ? "时序数据加载中..." : "时序数据尚未加载，进入时序数据或方案概览时会自动加载。"}</div>`;
    return;
  }
  const rows = state.payload.time_series || [];
  const [label, start, end] = monthRanges[state.month] || monthRanges[0];
  const pageRows = rows.slice(start, end);
  document.getElementById("pageInfo").textContent = `${label} 第 ${start + 1}-${Math.min(end, rows.length)} 小时`;
  const fields = ["datetime", "wind_speed", "solar_irradiance", "temperature", "load"];
  container.innerHTML = `<table><thead><tr><th>小时序号</th><th>时间</th><th>风速</th><th>太阳辐照</th><th>温度</th><th>负荷</th></tr></thead><tbody>${pageRows
    .map((row, offset) => {
      const index = start + offset;
      return `<tr><td>${row.hour_index}</td>${fields
        .map((key) => `<td><input data-time-index="${index}" data-key="${key}" value="${escapeHtml(row[key])}"></td>`)
        .join("")}</tr>`;
    })
    .join("")}</tbody></table>`;
  container.querySelectorAll("input").forEach((input) => input.addEventListener("input", onTimeInput));
}

function onTimeInput(event) {
  const input = event.target;
  const row = state.payload.time_series[Number(input.dataset.timeIndex)];
  row[input.dataset.key] = coerceInput(input.value);
  renderChart();
  renderLimitSummary();
}

function renderDeviceTables() {
  const jump = document.getElementById("deviceJump");
  const host = document.getElementById("deviceTables");
  if (!state.payload) {
    jump.innerHTML = "";
    host.innerHTML = "";
    return;
  }
  const shownSpecs = deviceSpecs.filter(([key]) => visibleDevices.has(key));
  jump.innerHTML = shownSpecs.map(([key, title]) => `<a href="#${key}">${title}</a>`).join("");
  if (!shownSpecs.length) {
    host.innerHTML = "<div class=\"validation-item\">当前未选择任何设备类型。</div>";
    return;
  }
  host.innerHTML = shownSpecs
    .map(([key, title, fields]) => `<section id="${key}" class="device-card"><div class="panel-heading"><h2>${title}</h2><button class="add-row" type="button" data-device="${key}">新增行</button></div>${deviceTable(key, fields)}</section>`)
    .join("");
  host.querySelectorAll("input").forEach((input) => input.addEventListener("input", onDeviceInput));
  host.querySelectorAll(".delete-row").forEach((button) => button.addEventListener("click", deleteDeviceRow));
  host.querySelectorAll(".add-row").forEach((button) => button.addEventListener("click", addDeviceRow));
}

function renderDeviceFilters() {
  const host = document.getElementById("deviceFilters");
  if (!host) return;
  host.innerHTML = deviceGroups
    .map(([key, title, devices]) => `<label class="device-filter"><input type="checkbox" data-device-group="${key}" ${devices.every((device) => visibleDevices.has(device)) ? "checked" : ""}> ${title}</label>`)
    .join("");
  host.querySelectorAll("[data-device-group]").forEach((input) => {
    const group = deviceGroups.find(([key]) => key === input.dataset.deviceGroup);
    const devices = group ? group[2] : [];
    input.indeterminate = devices.some((device) => visibleDevices.has(device)) && !devices.every((device) => visibleDevices.has(device));
    input.addEventListener("change", () => {
      devices.forEach((device) => {
        if (input.checked) {
          visibleDevices.add(device);
        } else {
          visibleDevices.delete(device);
        }
      });
      renderDeviceFilters();
      renderDeviceTables();
    });
  });
}

function deviceTable(key, fields) {
  const rows = state.payload[key] || [];
  return `<div class="data-table"><table><thead><tr>${fields.map((field) => `<th>${labels[field] || field}</th>`).join("")}<th>操作</th></tr></thead><tbody>${rows
    .map((row, index) => `<tr>${fields.map((field) => `<td><input data-device="${key}" data-row="${index}" data-key="${field}" value="${escapeHtml(row[field])}"></td>`).join("")}<td><button class="delete-row" type="button" data-device="${key}" data-row="${index}">删除</button></td></tr>`)
    .join("")}</tbody></table></div>`;
}

function onDeviceInput(event) {
  const input = event.target;
  state.payload[input.dataset.device][Number(input.dataset.row)][input.dataset.key] = coerceInput(input.value);
  renderLimitSummary();
  renderSummary();
}

function addDeviceRow(event) {
  const key = event.target.dataset.device;
  const spec = deviceSpecs.find((item) => item[0] === key);
  const row = Object.fromEntries(spec[2].map((field) => [field, field === "name" ? `${spec[1]}${(state.payload[key] || []).length + 1}` : 0]));
  state.payload[key] = state.payload[key] || [];
  state.payload[key].push(row);
  renderDeviceTables();
  renderLimitSummary();
  renderSummary();
}

function deleteDeviceRow(event) {
  state.payload[event.target.dataset.device].splice(Number(event.target.dataset.row), 1);
  renderDeviceTables();
  renderLimitSummary();
  renderSummary();
}

function renderPlanningParameters() {
  const host = document.getElementById("planningParametersTable");
  if (!host) return;
  if (!state.payload) {
    host.innerHTML = "";
    return;
  }
  const row = planningParameterRow();
  host.innerHTML = `<table><thead><tr><th>参数名称</th><th>参数值</th><th>取值范围</th></tr></thead><tbody>${planningParameterSpecs
    .map(([key, label, type, options]) => `<tr><td>${label}</td><td>${planningParameterControl(key, type, options, row[key])}</td><td>${planningParameterRangeText(type, options)}</td></tr>`)
    .join("")}</tbody></table>`;
  host.querySelectorAll("[data-planning-key]").forEach((input) => {
    const eventName = input.tagName === "SELECT" || input.type === "checkbox" ? "change" : "input";
    input.addEventListener(eventName, onPlanningParameterInput);
  });
}

function planningParameterControl(key, type, options, value) {
  if (type === "boolean") {
    const checked = truthyPlanningValue(value);
    return `<select class="planning-bool-select" data-planning-key="${key}" data-planning-type="boolean"><option value="true" ${checked ? "selected" : ""}>是</option><option value="false" ${checked ? "" : "selected"}>否</option></select>`;
  }
  const attrs = [
    `data-planning-key="${key}"`,
    'type="number"',
    options.min !== undefined ? `min="${options.min}"` : "",
    options.max !== undefined ? `max="${options.max}"` : "",
    `step="${options.integer ? 1 : 0.01}"`,
  ].filter(Boolean).join(" ");
  return `<input ${attrs} value="${escapeHtml(value)}">`;
}

function onPlanningParameterInput(event) {
  const input = event.target;
  const row = planningParameterRow();
  row[input.dataset.planningKey] = input.dataset.planningType === "boolean" ? truthyPlanningValue(input.value) : input.type === "checkbox" ? input.checked : coerceInput(input.value);
  renderLimitSummary();
  renderSummary();
}

function planningParameterRow() {
  if (!state.payload) return defaultPlanningParameterRow();
  if (!Array.isArray(state.payload.planning_parameters)) {
    state.payload.planning_parameters = state.payload.planning_parameters ? [state.payload.planning_parameters] : [defaultPlanningParameterRow()];
  }
  if (!state.payload.planning_parameters.length) {
    state.payload.planning_parameters.push(defaultPlanningParameterRow());
  }
  state.payload.planning_parameters[0] = normalizePlanningParameterRow(state.payload.planning_parameters[0]);
  return state.payload.planning_parameters[0];
}

function defaultPlanningParameterRow() {
  return Object.fromEntries(planningParameterSpecs.map(([key, , , options]) => [key, options.defaultValue]));
}

function normalizePlanningParameterRow(row) {
  const normalized = { ...defaultPlanningParameterRow(), ...(row || {}) };
  planningParameterSpecs.forEach(([key, , type]) => {
    if (type === "boolean") {
      normalized[key] = truthyPlanningValue(normalized[key]);
    }
  });
  return normalized;
}

function renderPlanningParameterSummaryTable() {
  if (!state.payload) return "";
  const row = planningParameterRow();
  return `<table><thead><tr><th>参数名称</th><th>参数值</th><th>取值范围</th></tr></thead><tbody>${planningParameterSpecs
    .map(([key, label, type, options]) => `<tr><td>${label}</td><td>${escapeHtml(formatPlanningParameterValue(row[key], type))}</td><td>${planningParameterRangeText(type, options)}</td></tr>`)
    .join("")}</tbody></table>`;
}

function formatPlanningParameterValue(value, type) {
  if (type === "boolean") return truthyPlanningValue(value) ? "是" : "否";
  return value;
}

function planningParameterRangeText(type, options) {
  if (type === "boolean") return "是/否";
  if (options.min !== undefined && options.max !== undefined) return `${options.min} - ${options.max}`;
  if (options.min !== undefined) return `不小于 ${options.min}`;
  if (options.max !== undefined) return `不大于 ${options.max}`;
  return "-";
}

function truthyPlanningValue(value) {
  if (typeof value === "boolean") return value;
  if (typeof value === "number") return value !== 0;
  return ["true", "1", "yes", "y", "是"].includes(String(value).trim().toLowerCase());
}

function renderLimitSummary() {
  renderSchemeSummary();
}

function renderSchemeSummary() {
  const hosts = [
    document.getElementById("summaryCharts"),
    document.getElementById("quantitySummary"),
    document.getElementById("planningSummary"),
  ].filter(Boolean);
  if (!state.payload) {
    hosts.forEach((host) => {
      host.innerHTML = "";
    });
    return;
  }

  const rows = isTimeSeriesLoaded() ? state.payload.time_series || [] : [];
  const chartsHost = document.getElementById("summaryCharts");
  const quantityHost = document.getElementById("quantitySummary");
  const planningHost = document.getElementById("planningSummary");

  if (chartsHost) {
    chartsHost.innerHTML = isTimeSeriesLoaded()
      ? summarySeries.map(([key, title, color, unit]) => renderHistogramPanel(rows, key, title, color, unit)).join("")
      : renderTimeSeriesPlaceholder("加载后显示风速、太阳辐照、温度、负荷直方图。");
  }

  if (quantityHost) {
    quantityHost.innerHTML = renderCandidateDeviceTable();
  }

  if (planningHost) {
    planningHost.innerHTML = renderPlanningParameterSummaryTable();
  }
  requestAnimationFrame(syncAdaptiveLayout);
}

function renderHistogramPanel(rows, key, title, color, unit) {
  const values = numericValues(rows, key);
  const stats = calculateSeriesStats(rows, key);
  return `<div class="histogram-panel"><div class="histogram-head"><strong>${title}分布</strong><span>${stats.count}点</span></div>${histogramSvg(values, color, title)}<div class="histogram-meta">最小值 ${formatNumber(stats.min)} ${unit} / 最大值 ${formatNumber(stats.max)} ${unit} / 平均值 ${formatNumber(stats.avg)} ${unit}</div></div>`;
}

function renderTimeSeriesPlaceholder(message) {
  return `<div class="empty-summary">时序数据尚未加载，${state.timeSeriesLoading ? "正在自动加载。" : message}</div>`;
}

function renderCandidateDeviceTable() {
  const rows = deviceSpecs.flatMap(([key, title]) =>
    (state.payload[key] || []).map((row) => ({
      device: title,
      name: row.name,
      capacity: capacityValue(key, row),
      lower: row.quantity_lower,
      upper: row.quantity_upper,
    })),
  );
  if (!rows.length) {
    return "<div class=\"empty-summary\">暂无设备条目</div>";
  }
  return `<table><thead><tr><th>设备类型</th><th>名称</th><th>容量</th><th>数据下限(台)</th><th>数据上限(台)</th><th>状态</th></tr></thead><tbody>${rows
    .map((row) => {
      const status = limitStatus(row.lower, row.upper);
      return `<tr><td>${row.device}</td><td>${escapeHtml(row.name)}</td><td>${escapeHtml(row.capacity)}</td><td>${escapeHtml(row.lower)}</td><td>${escapeHtml(row.upper)}</td><td class="${status === "正常" ? "status-ok" : "status-error"}">${status}</td></tr>`;
    })
    .join("")}</tbody></table>`;
}

function capacityValue(key, row) {
  const fieldByDevice = {
    diesel_generators: "capacity",
    wind_turbines: "capacity",
    photovoltaics: "capacity",
    storage_pcs: "power_capacity",
    storage_battery_packs: "battery_capacity",
    hydrogen_electrolyzers: "power_capacity",
    hydrogen_tanks: "hydrogen_tank_capacity",
    fuel_cells: "power_capacity",
  };
  return row[fieldByDevice[key]] ?? "";
}

function limitStatus(lower, upper) {
  if (lower === "" || upper === "") return "未填写";
  const lowerNumber = Number(lower);
  const upperNumber = Number(upper);
  if (!Number.isFinite(lowerNumber) || !Number.isFinite(upperNumber)) return "错误";
  return upperNumber < lowerNumber ? "错误" : "正常";
}

function numericValues(rows, key) {
  return rows
    .map((row) => row[key])
    .filter((value) => value !== "" && value !== null && value !== undefined)
    .map((value) => Number(value))
    .filter((value) => Number.isFinite(value));
}

function calculateSeriesStats(rows, key) {
  const values = numericValues(rows, key);
  if (!values.length) {
    return { count: 0, min: null, max: null, avg: null };
  }
  const min = Math.min(...values);
  const max = Math.max(...values);
  const avg = values.reduce((sum, value) => sum + value, 0) / values.length;
  return { count: values.length, min, max, avg };
}

function buildHistogram(values, binCount = 12) {
  const cleanValues = values.filter((value) => Number.isFinite(value));
  if (!cleanValues.length) return [];
  const min = Math.min(...cleanValues);
  const max = Math.max(...cleanValues);
  if (min === max) return [{ lower: min, upper: max, count: cleanValues.length }];
  const step = (max - min) / binCount;
  const bins = Array.from({ length: binCount }, (_, index) => ({
    lower: min + step * index,
    upper: index === binCount - 1 ? max : min + step * (index + 1),
    count: 0,
  }));
  cleanValues.forEach((value) => {
    const index = Math.min(binCount - 1, Math.floor((value - min) / step));
    bins[index].count += 1;
  });
  return bins;
}

function histogramSvg(values, color, title = "") {
  const bins = buildHistogram(values);
  if (!bins.length) {
    return "<div class=\"empty-summary\">暂无时序数据</div>";
  }
  const width = 460;
  const height = 220;
  const padLeft = 48;
  const padRight = 20;
  const padTop = 34;
  const padBottom = 34;
  const plotWidth = width - padLeft - padRight;
  const plotHeight = height - padTop - padBottom;
  const step = plotWidth / bins.length;
  const barWidth = Math.max(2, step - 4);
  const maxCount = Math.max(1, ...bins.map((bin) => bin.count));
  const y = (count) => padTop + plotHeight - (count / maxCount) * plotHeight;
  const yTicks = [0, 0.5, 1].map((ratio) => maxCount * ratio);
  const yAxis = yTicks
    .map((count) => {
      const tickY = y(count);
      return `<line x1="${padLeft}" x2="${width - padRight}" y1="${tickY.toFixed(1)}" y2="${tickY.toFixed(1)}" stroke="#d4e1dd"/><text x="${padLeft - 8}" y="${(tickY + 4).toFixed(1)}" fill="#5a716e" font-size="11" text-anchor="end">${formatInteger(count)}</text>`;
    })
    .join("");
  const bars = bins
    .map((bin, index) => {
      const barHeight = (bin.count / maxCount) * plotHeight;
      const x = padLeft + step * index + (step - barWidth) / 2;
      const barY = padTop + plotHeight - barHeight;
      const title = `${formatNumber(bin.lower)} - ${formatNumber(bin.upper)}: ${bin.count}`;
      const labelY = Math.max(12, barY - 5);
      return `<rect class="histogram-bar" data-bin-range="${escapeHtml(formatHistogramRange(bin))}" data-bin-count="${escapeHtml(formatInteger(bin.count))}" x="${x.toFixed(1)}" y="${barY.toFixed(1)}" width="${barWidth.toFixed(1)}" height="${barHeight.toFixed(1)}" rx="3" fill="${color}"><title>${escapeHtml(title)}</title></rect><text x="${(x + barWidth / 2).toFixed(1)}" y="${labelY.toFixed(1)}" fill="#294944" font-size="10" text-anchor="middle">${formatInteger(bin.count)}</text>`;
    })
    .join("");
  const minLabel = formatNumber(bins[0].lower);
  const maxLabel = formatNumber(bins[bins.length - 1].upper);
  return `<svg class="histogram-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeHtml(title)}统计直方图"><g>${yAxis}</g><line x1="${padLeft}" x2="${width - padRight}" y1="${padTop + plotHeight}" y2="${padTop + plotHeight}" stroke="#8ba49f"/><line x1="${padLeft}" x2="${padLeft}" y1="${padTop}" y2="${padTop + plotHeight}" stroke="#8ba49f"/>${bars}<text x="${padLeft}" y="${height - 8}" fill="#5a716e" font-size="12">${escapeHtml(minLabel)}</text><text x="${width - padRight}" y="${height - 8}" fill="#5a716e" font-size="12" text-anchor="end">${escapeHtml(maxLabel)}</text></svg>`;
}

function formatNumber(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "-";
  return number.toLocaleString("zh-CN", { maximumFractionDigits: 2 });
}

function formatInteger(value) {
  const number = Number(value);
  return Number.isFinite(number) ? String(Math.round(number)) : "-";
}

function formatHistogramRange(bin) {
  return `${formatNumber(bin.lower)} - ${formatNumber(bin.upper)}`;
}

function onHistogramMouseMove(event) {
  const bar = event.target.closest?.(".histogram-bar");
  if (!bar) {
    hideHistogramTip();
    return;
  }
  const tip = document.getElementById("chartTip");
  if (!tip) return;
  tip.hidden = false;
  tip.innerHTML = `横坐标：${escapeHtml(bar.dataset.binRange || "")}<br>纵坐标：${escapeHtml(bar.dataset.binCount || "")}`;
  tip.style.left = `${Math.min(window.innerWidth - 200, event.clientX + 14)}px`;
  tip.style.top = `${Math.max(12, event.clientY - 28)}px`;
}

function hideHistogramTip() {
  const tip = document.getElementById("chartTip");
  if (tip && !document.getElementById("chartCursor")?.hidden) return;
  if (tip) tip.hidden = true;
}

function renderSummary() {
  const box = document.getElementById("schemeSummary");
  const list = document.getElementById("validationList");
  const currentSchemeName = document.getElementById("currentSchemeName");
  if (!state.payload) {
    if (currentSchemeName) currentSchemeName.textContent = "未选择方案";
    box.innerHTML = "未选择方案";
    list.innerHTML = "";
    return;
  }
  if (currentSchemeName) currentSchemeName.textContent = state.currentScheme;
  const timeSeriesCount = isTimeSeriesLoaded() ? (state.payload.time_series || []).length : state.payload.time_series_count || 0;
  box.innerHTML = `<div>当前方案：<strong>${escapeHtml(state.currentScheme)}</strong></div><div>时序行数：${timeSeriesCount}</div><div>设备条目：${deviceSpecs.reduce((sum, [key]) => sum + (state.payload[key] || []).length, 0)}</div>`;
  const localMessages = validateLocal();
  list.innerHTML = localMessages.map((item) => `<div class="validation-item ${item.level}">${escapeHtml(item.message)}</div>`).join("");
}

function validateLocal() {
  const messages = [...(state.payload.validation || []), ...collectSaveWarnings()];
  if (messages.some((item) => item.level === "error")) {
    return messages;
  }
  return messages.length ? messages : [{ level: "ok", message: "当前数据通过基础校验" }];
}

function collectSaveWarnings() {
  if (!state.payload) return [];
  const messages = [];
  if (!isTimeSeriesLoaded()) {
    if (Number(state.payload.time_series_count || 0) !== 8760) {
      messages.push({ level: "error", message: `时序数据行数应为8760，当前为${state.payload.time_series_count || 0}` });
    }
  } else if ((state.payload.time_series || []).length !== 8760) {
    messages.push({ level: "error", message: `时序数据行数应为8760，当前为${(state.payload.time_series || []).length}` });
  }
  deviceSpecs.forEach(([key, title]) => {
    (state.payload[key] || []).forEach((row, index) => {
      const quantityLower = Number(row.quantity_lower);
      const quantityUpper = Number(row.quantity_upper);
      if (!Number.isFinite(quantityLower) || !Number.isFinite(quantityUpper)) {
        messages.push({ level: "error", message: `${title}第${index + 1}行数据上下限必须为数值` });
      } else if (quantityUpper < quantityLower) {
        messages.push({ level: "error", message: `${title}第${index + 1}行数据上限不能小于数据下限` });
      }
    });
  });
  messages.push(...collectPlanningParameterWarnings());
  return messages;
}

function collectPlanningParameterWarnings() {
  if (!state.payload) return [];
  const row = planningParameterRow();
  const messages = [];
  planningParameterSpecs.forEach(([key, label, type, options]) => {
    if (type === "boolean") return;
    const value = Number(row[key]);
    if (!Number.isFinite(value)) {
      messages.push({ level: "error", message: `${label}必须为数值` });
      return;
    }
    if (options.integer && !Number.isInteger(value)) {
      messages.push({ level: "error", message: `${label}必须为整数` });
    }
    if (options.min !== undefined && value < options.min) {
      messages.push({ level: "error", message: `${label}不能小于${options.min}` });
    }
    if (options.max !== undefined && value > options.max) {
      messages.push({ level: "error", message: `${label}不能大于${options.max}` });
    }
  });
  const upper = Number(row.frequency_security_upper);
  const lower = Number(row.frequency_security_lower);
  if (Number.isFinite(upper) && Number.isFinite(lower) && upper < lower) {
    messages.push({ level: "error", message: "频率安全上限不能小于频率安全下限" });
  }
  return messages;
}

function coerceInput(value) {
  const text = String(value);
  if (text.trim() === "") return "";
  const number = Number(text);
  return Number.isFinite(number) ? number : text;
}

function normalizeSchemeName(name) {
  return String(name || "").replace(/[\s\u0000-\u001f\u007f\u200b-\u200f\u202a-\u202e\ufeff]/g, "");
}

function schemeNameExists(name, excludedName = "") {
  const cleanName = normalizeSchemeName(name);
  const cleanExcludedName = normalizeSchemeName(excludedName);
  return state.schemes.some((scheme) => normalizeSchemeName(scheme.name) === cleanName && normalizeSchemeName(scheme.name) !== cleanExcludedName);
}

function normalizePayload(payload) {
  if (!payload) return payload;
  payload.timeSeriesLoaded = Boolean(payload.time_series_loaded || payload.timeSeriesLoaded || payload.time_series);
  if (payload.time_series && payload.time_series_count === undefined) {
    payload.time_series_count = payload.time_series.length;
  }
  if (!Array.isArray(payload.planning_parameters)) {
    payload.planning_parameters = payload.planning_parameters ? [payload.planning_parameters] : [defaultPlanningParameterRow()];
  }
  payload.planning_parameters[0] = normalizePlanningParameterRow(payload.planning_parameters[0]);
  return payload;
}

function isTimeSeriesLoaded() {
  return Boolean(state.payload && state.payload.timeSeriesLoaded);
}

function setTimeSeriesLoaded(value) {
  if (!state.payload) return;
  state.payload.timeSeriesLoaded = value;
  state.payload.time_series_loaded = value;
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[char]);
}

function showError(error) {
  alert(error.message || String(error));
  return null;
}
