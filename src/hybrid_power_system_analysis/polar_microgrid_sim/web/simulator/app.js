const apiBase = (window.POLAR_SIM_API_URL || localStorage.getItem("polarSimApiUrl") || location.origin).replace(/\/$/, "");
const state = {
  snapshot: null,
  deviceFaults: [],
  measurementFaults: [],
  modes: [],
  weatherPoints: [],
  loadPoints: [],
};

const $ = (id) => document.getElementById(id);

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

function generateCurves(jitter = 0) {
  const windPeak = Number($("windPeak").value) + jitter;
  const solarPeak = Number($("solarPeak").value);
  const tempMean = Number($("tempMean").value);
  const loadBase = Number($("loadBase").value);
  const weather = [];
  const loads = [];
  for (let i = 0; i < 96; i += 1) {
    const minute = i * 15;
    const day = minute / 1440;
    const gust = Math.sin(day * Math.PI * 2 * 5 + 0.8) * 4 + Math.sin(day * Math.PI * 2 * 11) * 2;
    const wind = Math.max(0, Math.min(50, windPeak * (0.58 + 0.28 * Math.sin(day * Math.PI * 2 - 0.7)) + gust));
    const sunShape = Math.max(0, Math.sin((day - 0.25) * Math.PI * 2));
    const temp = tempMean + 6 * Math.sin((day - 0.33) * Math.PI * 2);
    const load = loadBase * (0.84 + 0.18 * Math.sin((day - 0.18) * Math.PI * 2) + 0.08 * Math.sin(day * Math.PI * 8));
    weather.push({
      minute,
      wind_speed_mps: Number(wind.toFixed(2)),
      air_temp_c: Number(temp.toFixed(2)),
      air_pressure_hpa: Number((955 + 10 * Math.sin(day * Math.PI * 2 + 0.4)).toFixed(2)),
      solar_irradiance_w_m2: Number((solarPeak * sunShape).toFixed(2)),
      humidity_pct: Number((68 + 9 * Math.sin(day * Math.PI * 2 + 2.2)).toFixed(2)),
    });
    loads.push({ minute, p_kw: Number(Math.max(20, load).toFixed(2)) });
  }
  state.weatherPoints = weather;
  state.loadPoints = loads;
  drawCurves();
}

function drawLineChart(canvas, series, colors, labels) {
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#fcfeff";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "#d8e1e5";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i += 1) {
    const y = 24 + i * ((height - 48) / 4);
    ctx.beginPath();
    ctx.moveTo(42, y);
    ctx.lineTo(width - 18, y);
    ctx.stroke();
  }
  const all = series.flatMap((item) => item.values);
  const min = Math.min(...all, 0);
  const max = Math.max(...all, 1);
  series.forEach((item, idx) => {
    ctx.strokeStyle = colors[idx];
    ctx.lineWidth = 2;
    ctx.beginPath();
    item.values.forEach((value, point) => {
      const x = 42 + (point / Math.max(1, item.values.length - 1)) * (width - 62);
      const y = height - 24 - ((value - min) / Math.max(1e-6, max - min)) * (height - 54);
      if (point === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  });
  ctx.fillStyle = "#172026";
  ctx.font = "12px Microsoft YaHei, Arial";
  labels.forEach((label, idx) => {
    ctx.fillStyle = colors[idx];
    ctx.fillRect(52 + idx * 96, 14, 18, 3);
    ctx.fillStyle = "#63717a";
    ctx.fillText(label, 76 + idx * 96, 18);
  });
}

function drawCurves() {
  const weather = state.weatherPoints.length ? state.weatherPoints : [];
  const loads = state.loadPoints.length ? state.loadPoints : [];
  drawLineChart(
    $("weatherChart"),
    [
      { values: weather.map((p) => p.wind_speed_mps || 0) },
      { values: weather.map((p) => p.solar_irradiance_w_m2 / 25 || 0) },
      { values: weather.map((p) => (p.air_temp_c || 0) + 50) },
    ],
    ["#008c8c", "#b87500", "#2b6b7f"],
    ["风速", "辐照/25", "气温+50"],
  );
  drawLineChart($("loadChart"), [{ values: loads.map((p) => p.p_kw || 0) }], ["#c93a3a"], ["负荷kW"]);
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
  renderCommands(snapshot.commands.history || []);
  renderDevices(snapshot.devices || []);
  if (!state.modes.length) {
    state.modes = (snapshot.devices || [])
      .filter((dev) => dev.mode !== undefined && dev.mode !== "")
      .slice(0, 8)
      .map((dev) => ({ dev_type: dev.dev_type, dev_name: dev.dev_name, mode: dev.mode || "PQ" }));
    renderModes();
  }
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

function renderFaults() {
  $("deviceFaults").innerHTML = state.deviceFaults.map((item, idx) => `
    <div class="fault-item">
      <strong>${item.dev_type}.${item.dev_name}</strong>
      <span>${item.start_minute} - ${item.clear_minute} min，run=${item.run_stat}，status=${item.status}</span>
      <button data-remove-device="${idx}">删除</button>
    </div>
  `).join("") || '<div class="fault-item"><span>暂无设备故障</span></div>';
  $("measFaults").innerHTML = state.measurementFaults.map((item, idx) => `
    <div class="fault-item">
      <strong>${item.name}</strong>
      <span>${item.start_minute} - ${item.clear_minute} min，${item.fault_type}</span>
      <button data-remove-meas="${idx}">删除</button>
    </div>
  `).join("") || '<div class="fault-item"><span>暂无量测故障</span></div>';
}

function renderModes() {
  $("modeTable").innerHTML = state.modes.map((item, idx) => `
    <div class="mode-row">
      <span>${item.dev_type}.${item.dev_name}</span>
      <select data-mode-index="${idx}">
        ${["PQ", "PV", "PH", "V"].map((mode) => `<option value="${mode}" ${mode === item.mode ? "selected" : ""}>${mode}</option>`).join("")}
      </select>
    </div>
  `).join("") || '<div class="fault-item"><span>暂无可设模式设备</span></div>';
}

async function pushCurves() {
  generateCurves(0);
  await api("/api/curves", {
    method: "POST",
    body: JSON.stringify({ weather: state.weatherPoints, loads: { load_ac_1: state.loadPoints } }),
  });
  $("curveStatus").textContent = "已写入";
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

$("pushCurves").addEventListener("click", pushCurves);
$("randomCurves").addEventListener("click", () => {
  generateCurves(Math.random() * 8 - 4);
  $("curveStatus").textContent = "本地扰动";
});
$("addDeviceFault").addEventListener("click", async () => {
  state.deviceFaults.push({
    dev_type: $("faultDevType").value.trim(),
    dev_name: $("faultDevName").value.trim(),
    start_minute: Number($("faultStart").value),
    clear_minute: Number($("faultClear").value),
    run_stat: 0,
    status: 0,
  });
  renderFaults();
  await pushSettings();
});
$("addMeasFault").addEventListener("click", async () => {
  state.measurementFaults.push({
    name: $("measTarget").value.trim(),
    fault_type: $("measFaultType").value,
    start_minute: Number($("measStart").value),
    clear_minute: Number($("measClear").value),
    bias: 10,
  });
  renderFaults();
  await pushSettings();
});
$("pushModes").addEventListener("click", pushSettings);
document.addEventListener("change", (event) => {
  if (event.target.dataset.modeIndex !== undefined) {
    state.modes[Number(event.target.dataset.modeIndex)].mode = event.target.value;
  }
});
document.addEventListener("click", async (event) => {
  if (event.target.dataset.removeDevice !== undefined) {
    state.deviceFaults.splice(Number(event.target.dataset.removeDevice), 1);
    renderFaults();
    await pushSettings();
  }
  if (event.target.dataset.removeMeas !== undefined) {
    state.measurementFaults.splice(Number(event.target.dataset.removeMeas), 1);
    renderFaults();
    await pushSettings();
  }
});

initPageNavigation();
generateCurves(0);
renderFaults();
setInterval(refresh, 2000);
refresh();
