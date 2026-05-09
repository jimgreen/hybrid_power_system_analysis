const state = { schemes: [], currentScheme: "", payload: null, page: 0, pageSize: 168 };

const deviceSpecs = [
  ["diesel_generators", "柴发", ["name", "capacity", "design_capacity_lower", "design_capacity_upper", "cost", "power_upper", "power_lower", "fuel_rate"]],
  ["wind_turbines", "风机", ["name", "capacity", "design_capacity_lower", "design_capacity_upper", "cost", "cut_in_wind_speed", "cut_out_wind_speed"]],
  ["photovoltaics", "光伏", ["name", "capacity", "design_capacity_lower", "design_capacity_upper", "cost", "cut_in_wind_speed", "cut_out_wind_speed"]],
  ["storage_pcs", "储能PCS", ["name", "power_capacity", "design_capacity_lower", "design_capacity_upper", "cost"]],
  ["storage_battery_packs", "储能电池组", ["name", "battery_capacity", "design_capacity_lower", "design_capacity_upper", "cost"]],
  ["hydrogen_electrolyzers", "电制氢", ["name", "power_capacity", "design_capacity_lower", "design_capacity_upper", "cost", "electric_to_hydrogen_efficiency"]],
  ["hydrogen_tanks", "储氢罐", ["name", "hydrogen_tank_capacity", "design_capacity_lower", "design_capacity_upper", "cost"]],
  ["fuel_cells", "燃料电池", ["name", "power_capacity", "design_capacity_lower", "design_capacity_upper", "cost", "hydrogen_to_electric_efficiency"]],
];

const visibleDevices = new Set(deviceSpecs.map(([key]) => key));

const labels = {
  name: "名称",
  capacity: "容量",
  power_capacity: "功率容量",
  battery_capacity: "电池容量",
  hydrogen_tank_capacity: "储氢罐容量",
  design_capacity_lower: "设计容量下限",
  design_capacity_upper: "设计容量上限",
  cost: "成本",
  power_upper: "功率上限",
  power_lower: "功率下限",
  fuel_rate: "油耗率",
  cut_in_wind_speed: "切入风速",
  cut_out_wind_speed: "切出风速",
  electric_to_hydrogen_efficiency: "电-氢效率",
  hydrogen_to_electric_efficiency: "氢-电效率",
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
    });
  });
}

function bindActions() {
  document.getElementById("createScheme").addEventListener("click", createScheme);
  document.getElementById("copyScheme").addEventListener("click", copyScheme);
  document.getElementById("renameScheme").addEventListener("click", renameScheme);
  document.getElementById("saveScheme").addEventListener("click", saveScheme);
  document.getElementById("prevPage").addEventListener("click", () => {
    state.page = Math.max(0, state.page - 1);
    renderTimeTable();
  });
  document.getElementById("nextPage").addEventListener("click", () => {
    state.page += 1;
    renderTimeTable();
  });
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
  state.payload = await api(`/api/planning/schemes/${encodeURIComponent(name)}`);
  state.page = 0;
  renderAll();
}

async function createScheme() {
  const name = prompt("请输入新方案名称");
  if (!name) return;
  state.payload = await api("/api/planning/schemes", { method: "POST", body: JSON.stringify({ name }) }).catch(showError);
  if (!state.payload) return;
  state.currentScheme = state.payload.scheme;
  await loadSchemes();
  renderAll();
}

async function copyScheme() {
  if (!state.currentScheme) return alert("请先选择方案");
  const target = prompt("请输入复制后的方案名称", `${state.currentScheme}_副本`);
  if (!target) return;
  state.payload = await api("/api/planning/schemes/copy", {
    method: "POST",
    body: JSON.stringify({ source: state.currentScheme, target }),
  }).catch(showError);
  if (!state.payload) return;
  state.currentScheme = state.payload.scheme;
  await loadSchemes();
  renderAll();
}

async function renameScheme() {
  if (!state.currentScheme) return alert("请先选择方案");
  const target = prompt("请输入新的方案名称", state.currentScheme);
  if (!target || target === state.currentScheme) return;
  state.payload = await api("/api/planning/schemes/rename", {
    method: "POST",
    body: JSON.stringify({ source: state.currentScheme, target }),
  }).catch(showError);
  if (!state.payload) return;
  state.currentScheme = state.payload.scheme;
  await loadSchemes();
  renderAll();
}

async function saveScheme() {
  if (!state.currentScheme || !state.payload) return alert("请先选择方案");
  const warnings = collectSaveWarnings();
  if (warnings.length) {
    renderSummary();
    alert(`参数校验未通过：\n${warnings.map((item) => `- ${item.message}`).join("\n")}`);
    return;
  }
  state.payload = await api(`/api/planning/schemes/${encodeURIComponent(state.currentScheme)}`, {
    method: "PUT",
    body: JSON.stringify(state.payload),
  }).catch(showError);
  if (!state.payload) return;
  renderAll();
  alert("保存成功");
}

function renderAll() {
  renderSchemes();
  renderChart();
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
  const rows = state.payload.time_series || [];
  const width = svg.clientWidth || 900;
  const height = 320;
  const pad = 34;
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  const curves = [
    ["wind_speed", "#1f9bb4", "风速"],
    ["solar_irradiance", "#d79018", "太阳辐射"],
    ["load", "#2d6b45", "负荷"],
  ].filter(([key]) => document.querySelector(`[data-curve="${key}"]`).checked);
  const maxValue = Math.max(1, ...curves.flatMap(([key]) => rows.map((row) => Number(row[key]) || 0)));
  const x = (index) => pad + (index / Math.max(1, rows.length - 1)) * (width - pad * 2);
  const y = (value) => height - pad - ((Number(value) || 0) / maxValue) * (height - pad * 2);
  const grid = [0, 1, 2, 3]
    .map((n) => `<line x1="${pad}" x2="${width - pad}" y1="${pad + (n * (height - pad * 2)) / 3}" y2="${pad + (n * (height - pad * 2)) / 3}" stroke="#9db4ae"/>`)
    .join("");
  const paths = curves
    .map(([key, color]) => {
      const d = rows.map((row, index) => `${index === 0 ? "M" : "L"}${x(index).toFixed(1)},${y(row[key]).toFixed(1)}`).join(" ");
      return `<path d="${d}" fill="none" stroke="${color}" stroke-width="2" vector-effect="non-scaling-stroke"/>`;
    })
    .join("");
  svg.innerHTML = `<rect x="0" y="0" width="${width}" height="${height}" rx="18" fill="transparent"/><g opacity=".35">${grid}</g>${paths}`;
}

function renderTimeTable() {
  const container = document.getElementById("timeTable");
  if (!state.payload) {
    container.innerHTML = "";
    return;
  }
  const rows = state.payload.time_series || [];
  const totalPages = Math.max(1, Math.ceil(rows.length / state.pageSize));
  state.page = Math.min(state.page, totalPages - 1);
  const start = state.page * state.pageSize;
  const pageRows = rows.slice(start, start + state.pageSize);
  document.getElementById("pageInfo").textContent = `第 ${state.page + 1} / ${totalPages} 页`;
  container.innerHTML = `<table><thead><tr><th>小时序号</th><th>时间</th><th>风速</th><th>太阳辐射</th><th>负荷</th></tr></thead><tbody>${pageRows
    .map((row, offset) => {
      const index = start + offset;
      return `<tr><td>${row.hour_index}</td>${["datetime", "wind_speed", "solar_irradiance", "load"]
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
  host.innerHTML = deviceSpecs
    .map(([key, title]) => `<label class="device-filter"><input type="checkbox" data-device-filter="${key}" ${visibleDevices.has(key) ? "checked" : ""}> ${title}</label>`)
    .join("");
  host.querySelectorAll("[data-device-filter]").forEach((input) => {
    input.addEventListener("change", () => {
      if (input.checked) {
        visibleDevices.add(input.dataset.deviceFilter);
      } else {
        visibleDevices.delete(input.dataset.deviceFilter);
      }
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
  const host = document.getElementById("limitSummary");
  if (!state.payload) {
    host.innerHTML = "";
    return;
  }
  const rows = deviceSpecs.flatMap(([key, title]) =>
    (state.payload[key] || []).map((row) => ({
      device: title,
      name: row.name,
      lower: row.design_capacity_lower,
      upper: row.design_capacity_upper,
    })),
  );
  host.innerHTML = `<table><thead><tr><th>设备类型</th><th>名称</th><th>设计容量下限</th><th>设计容量上限</th><th>状态</th></tr></thead><tbody>${rows
    .map((row) => `<tr><td>${row.device}</td><td>${escapeHtml(row.name)}</td><td>${row.lower}</td><td>${row.upper}</td><td>${Number(row.lower) > Number(row.upper) ? "错误" : "正常"}</td></tr>`)
    .join("")}</tbody></table>`;
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
  box.innerHTML = `<div>当前方案：<strong>${escapeHtml(state.currentScheme)}</strong></div><div>8760行数：${(state.payload.time_series || []).length}</div><div>设备条目：${deviceSpecs.reduce((sum, [key]) => sum + (state.payload[key] || []).length, 0)}</div>`;
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
  if ((state.payload.time_series || []).length !== 8760) {
    messages.push({ level: "error", message: `8760时序数据行数应为8760，当前为${(state.payload.time_series || []).length}` });
  }
  deviceSpecs.forEach(([key, title]) => {
    (state.payload[key] || []).forEach((row, index) => {
      const lower = Number(row.design_capacity_lower);
      const upper = Number(row.design_capacity_upper);
      if (!Number.isFinite(lower) || !Number.isFinite(upper)) {
        messages.push({ level: "error", message: `${title}第${index + 1}行设计容量上下限必须为数值` });
      } else if (upper < lower) {
        messages.push({ level: "error", message: `${title}第${index + 1}行设计容量上限不能小于下限` });
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

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[char]);
}

function showError(error) {
  alert(error.message || String(error));
  return null;
}
