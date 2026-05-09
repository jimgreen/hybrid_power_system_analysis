const state = { schemes: [], currentScheme: "", payload: null, month: 0, timeSeriesLoading: null };

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
  bindActions();
  loadSchemes().catch(showError);
});

function bindTabs() {
  document.querySelectorAll(".tab").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((item) => item.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((item) => item.classList.remove("active"));
      button.classList.add("active");
      document.getElementById(`${button.dataset.tab}Tab`).classList.add("active");
      ensureTimeSeriesForActiveTab();
    });
  });
}

function bindActions() {
  document.getElementById("createScheme").addEventListener("click", createScheme);
  document.getElementById("copyScheme").addEventListener("click", copyScheme);
  document.getElementById("renameScheme").addEventListener("click", renameScheme);
  document.getElementById("saveScheme").addEventListener("click", saveScheme);
  document.getElementById("deleteScheme").addEventListener("click", deleteScheme);
  document.querySelectorAll("[data-curve]").forEach((box) => box.addEventListener("change", renderChart));
  window.addEventListener("resize", renderChart);
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
  renderLimitSummary();
  renderSummary();
}

function renderChart() {
  const svg = document.getElementById("timeChart");
  if (!state.payload) {
    svg.innerHTML = "";
    return;
  }
  if (!isTimeSeriesLoaded()) {
    const width = svg.clientWidth || 900;
    const height = 320;
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    svg.innerHTML = `<rect x="0" y="0" width="${width}" height="${height}" rx="18" fill="transparent"/><text x="${width / 2}" y="${height / 2}" text-anchor="middle" fill="#5a716e" font-size="16">${state.timeSeriesLoading ? "时序数据加载中..." : "时序数据尚未加载"}</text>`;
    return;
  }
  const rows = state.payload.time_series || [];
  const width = svg.clientWidth || 900;
  const height = 320;
  const pad = 34;
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  const curves = summarySeries.filter(([key]) => {
    const checkbox = document.querySelector(`[data-curve="${key}"]`);
    return checkbox && checkbox.checked;
  });
  const values = curves.flatMap(([key]) => numericValues(rows, key));
  const minValue = Math.min(0, ...values);
  const maxValue = Math.max(1, ...values);
  const valueSpan = maxValue - minValue || 1;
  const x = (index) => pad + (index / Math.max(1, rows.length - 1)) * (width - pad * 2);
  const y = (value) => {
    const number = Number(value);
    const safeValue = Number.isFinite(number) ? number : 0;
    return height - pad - ((safeValue - minValue) / valueSpan) * (height - pad * 2);
  };
  const grid = [0, 1, 2, 3]
    .map((n) => `<line x1="${pad}" x2="${width - pad}" y1="${pad + (n * (height - pad * 2)) / 3}" y2="${pad + (n * (height - pad * 2)) / 3}" stroke="#9db4ae"/>`)
    .join("");
  const paths = curves
    .map(([key, , color]) => {
      const d = rows.map((row, index) => `${index === 0 ? "M" : "L"}${x(index).toFixed(1)},${y(row[key]).toFixed(1)}`).join(" ");
      return `<path d="${d}" fill="none" stroke="${color}" stroke-width="2" vector-effect="non-scaling-stroke"/>`;
    })
    .join("");
  svg.innerHTML = `<rect x="0" y="0" width="${width}" height="${height}" rx="18" fill="transparent"/><g opacity=".35">${grid}</g>${paths}`;
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
    container.innerHTML = `<div class="empty-summary">${state.timeSeriesLoading ? "时序数据加载中..." : "时序数据尚未加载，进入8760时序数据或方案概览时会自动加载。"}</div>`;
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

function renderLimitSummary() {
  renderSchemeSummary();
}

function renderSchemeSummary() {
  const hosts = [
    document.getElementById("schemeOverview"),
    document.getElementById("summaryCharts"),
    document.getElementById("quantitySummary"),
  ].filter(Boolean);
  if (!state.payload) {
    hosts.forEach((host) => {
      host.innerHTML = "";
    });
    return;
  }

  const rows = isTimeSeriesLoaded() ? state.payload.time_series || [] : [];
  const timeSeriesCount = isTimeSeriesLoaded() ? rows.length : state.payload.time_series_count || 0;
  const deviceCount = deviceSpecs.reduce((sum, [key]) => sum + (state.payload[key] || []).length, 0);
  const warningCount = collectSaveWarnings().filter((item) => item.level === "error").length;
  const overviewHost = document.getElementById("schemeOverview");
  const chartsHost = document.getElementById("summaryCharts");
  const quantityHost = document.getElementById("quantitySummary");

  if (overviewHost) {
    overviewHost.innerHTML = [
      ["当前方案", state.currentScheme || state.payload.scheme || "未选择方案"],
      ["8760行数", timeSeriesCount],
      ["设备条目", deviceCount],
      ["校验问题", warningCount ? `${warningCount}项` : "0项"],
    ]
      .map(([label, value]) => `<div class="overview-item"><span>${label}</span><strong>${escapeHtml(value)}</strong></div>`)
      .join("");
  }

  if (chartsHost) {
    chartsHost.innerHTML = isTimeSeriesLoaded()
      ? summarySeries.map(([key, title, color, unit]) => renderHistogramPanel(rows, key, title, color, unit)).join("")
      : renderTimeSeriesPlaceholder("加载后显示风速、太阳辐照、温度、负荷直方图。");
  }

  if (quantityHost) {
    quantityHost.innerHTML = renderCandidateDeviceTable();
  }
}

function renderHistogramPanel(rows, key, title, color, unit) {
  const values = numericValues(rows, key);
  const stats = calculateSeriesStats(rows, key);
  return `<div class="histogram-panel"><div class="histogram-head"><strong>${title}分布</strong><span>${stats.count}点</span></div>${histogramSvg(values, color)}<div class="histogram-meta">最小值 ${formatNumber(stats.min)} ${unit} / 最大值 ${formatNumber(stats.max)} ${unit} / 平均值 ${formatNumber(stats.avg)} ${unit}</div></div>`;
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

function histogramSvg(values, color) {
  const bins = buildHistogram(values);
  if (!bins.length) {
    return "<div class=\"empty-summary\">暂无时序数据</div>";
  }
  const width = 420;
  const height = 180;
  const padX = 34;
  const padY = 24;
  const plotWidth = width - padX * 2;
  const plotHeight = height - padY * 2;
  const step = plotWidth / bins.length;
  const barWidth = Math.max(2, step - 4);
  const maxCount = Math.max(1, ...bins.map((bin) => bin.count));
  const bars = bins
    .map((bin, index) => {
      const barHeight = (bin.count / maxCount) * plotHeight;
      const x = padX + step * index + (step - barWidth) / 2;
      const y = height - padY - barHeight;
      const title = `${formatNumber(bin.lower)} - ${formatNumber(bin.upper)}: ${bin.count}`;
      return `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${barWidth.toFixed(1)}" height="${barHeight.toFixed(1)}" rx="3" fill="${color}"><title>${escapeHtml(title)}</title></rect>`;
    })
    .join("");
  const minLabel = formatNumber(bins[0].lower);
  const maxLabel = formatNumber(bins[bins.length - 1].upper);
  return `<svg class="histogram-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="统计直方图">${bars}<line x1="${padX}" x2="${width - padX}" y1="${height - padY}" y2="${height - padY}" stroke="#8ba49f"/><text x="${padX}" y="${height - 4}" fill="#5a716e" font-size="12">${escapeHtml(minLabel)}</text><text x="${width - padX}" y="${height - 4}" fill="#5a716e" font-size="12" text-anchor="end">${escapeHtml(maxLabel)}</text></svg>`;
}

function formatNumber(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "-";
  return number.toLocaleString("zh-CN", { maximumFractionDigits: 2 });
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
  box.innerHTML = `<div>当前方案：<strong>${escapeHtml(state.currentScheme)}</strong></div><div>8760行数：${timeSeriesCount}</div><div>设备条目：${deviceSpecs.reduce((sum, [key]) => sum + (state.payload[key] || []).length, 0)}</div>`;
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
      messages.push({ level: "error", message: `8760时序数据行数应为8760，当前为${state.payload.time_series_count || 0}` });
    }
  } else if ((state.payload.time_series || []).length !== 8760) {
    messages.push({ level: "error", message: `8760时序数据行数应为8760，当前为${(state.payload.time_series || []).length}` });
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
