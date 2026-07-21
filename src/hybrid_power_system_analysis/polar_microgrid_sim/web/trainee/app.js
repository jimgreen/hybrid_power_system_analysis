const apiBase = (window.POLAR_SIM_API_URL || localStorage.getItem("polarSimApiUrl") || location.origin).replace(/\/$/, "");
const pending = { run_status: new Map(), set_values: new Map() };
const trend = [];

const $ = (id) => document.getElementById(id);

async function api(path, options = {}) {
  const response = await fetch(`${apiBase}${path}`, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

async function refresh() {
  try {
    const snapshot = await api("/api/snapshot");
    $("connectionDot").className = "ok";
    $("connectionText").textContent = "在线";
    renderSnapshot(snapshot);
  } catch (error) {
    $("connectionDot").className = "off";
    $("connectionText").textContent = "离线";
  }
}

function renderSnapshot(snapshot) {
  $("simTime").textContent = snapshot.clock.time;
  $("simState").textContent = snapshot.clock.state;
  const scada = snapshot.measurements.scada || [];
  $("measureCount").textContent = `${scada.length} 点`;
  $("validCount").textContent = `${scada.filter((m) => m.valid === 1).length} 可用`;
  renderMeasurements(scada);
  renderDevices(snapshot.devices || []);
  renderHistory(snapshot.commands.history || []);
  pushTrend(scada, snapshot.clock.time);
  drawTrend();
}

function renderMeasurements(items) {
  $("measurementTable").innerHTML = `
    <table>
      <thead><tr><th>量测</th><th>设备</th><th>类型</th><th>值</th><th>可用</th></tr></thead>
      <tbody>
        ${items.map((item) => {
          const valueClass = Math.abs(item.value || 0) > 10000 ? "value-bad" : Math.abs(item.value || 0) > 1000 ? "value-warn" : "";
          return `<tr>
            <td>${item.name}</td>
            <td>${item.dev_name}</td>
            <td>${item.meas_type}</td>
            <td class="${valueClass}">${formatNumber(item.value)}</td>
            <td>${item.valid ? "可用" : "停用"}</td>
          </tr>`;
        }).join("")}
      </tbody>
    </table>`;
}

function renderDevices(devices) {
  $("deviceControls").innerHTML = devices.slice(0, 28).map((dev) => {
    const key = `${dev.dev_type}|${dev.dev_name}`;
    const currentRun = pending.run_status.has(key) ? pending.run_status.get(key).run_stat : dev.run_stat;
    const setTypes = preferredSetTypes(dev);
    return `
      <div class="device-row">
        <div>
          <div class="device-name">${dev.dev_name}</div>
          <div class="device-type">${dev.dev_type} · ${dev.mode || "--"}</div>
        </div>
        <label class="toggle">
          <span>${currentRun ? "投入" : "退出"}</span>
          <input type="checkbox" data-run-key="${key}" ${currentRun ? "checked" : ""} />
        </label>
        <div class="setpoints">
          ${setTypes.map((setType) => `
            <label>${setType}
              <input type="number" step="0.01" data-set-key="${key}" data-set-type="${setType}" value="${currentSetValue(dev, setType)}" />
            </label>
          `).join("")}
        </div>
      </div>`;
  }).join("") || '<div class="log-item">暂无设备</div>';
}

function preferredSetTypes(dev) {
  const types = new Set(dev.set_types || []);
  const selected = [];
  if (types.has("p_set") || types.has("p_ac_set") || types.has("pv0")) selected.push("p_set");
  if (types.has("q_set") || types.has("q_ac_set") || types.has("qv0")) selected.push("q_set");
  if (types.has("v_set") || types.has("v_ac_set")) selected.push("v_set");
  return selected.slice(0, 3);
}

function currentSetValue(dev, setType) {
  const exact = dev.set_values?.[setType];
  if (exact !== undefined) return exact;
  const raw = dev.raw || {};
  if (setType === "p_set") return raw.p_set ?? raw.p_ac_set ?? raw.pv0 ?? "";
  if (setType === "q_set") return raw.q_set ?? raw.q_ac_set ?? raw.qv0 ?? "";
  if (setType === "v_set") return raw.v_set ?? raw.v_ac_set ?? "";
  return "";
}

function renderHistory(history) {
  $("commandHistory").innerHTML = history.slice(-9).reverse().map((item) => `
    <div class="log-item">
      <strong>${item.time || ""}</strong>
      <span>${item.source || "student"} · 投退 ${item.accepted?.run_status || 0} · 设值 ${item.accepted?.set_values || 0}</span>
    </div>
  `).join("") || '<div class="log-item">暂无记录</div>';
}

function pushTrend(scada, time) {
  const firstValues = scada.slice(0, 5).map((item) => Number(item.value || 0));
  trend.push({ time, values: firstValues });
  while (trend.length > 120) trend.shift();
}

function drawTrend() {
  const canvas = $("trendChart");
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#fcfeff";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "#d8e2e6";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i += 1) {
    const y = 24 + i * ((height - 48) / 4);
    ctx.beginPath();
    ctx.moveTo(44, y);
    ctx.lineTo(width - 20, y);
    ctx.stroke();
  }
  const values = trend.flatMap((item) => item.values);
  const min = Math.min(...values, -1);
  const max = Math.max(...values, 1);
  const colors = ["#008c8c", "#b87500", "#2b6b7f", "#c93a3a", "#23854a"];
  for (let seriesIndex = 0; seriesIndex < 5; seriesIndex += 1) {
    ctx.strokeStyle = colors[seriesIndex];
    ctx.lineWidth = 2;
    ctx.beginPath();
    trend.forEach((item, idx) => {
      const value = item.values[seriesIndex] || 0;
      const x = 44 + (idx / Math.max(1, trend.length - 1)) * (width - 68);
      const y = height - 24 - ((value - min) / Math.max(1e-6, max - min)) * (height - 56);
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }
  ctx.font = "12px Microsoft YaHei, Arial";
  colors.forEach((color, idx) => {
    ctx.fillStyle = color;
    ctx.fillRect(54 + idx * 82, 14, 18, 3);
    ctx.fillStyle = "#64737c";
    ctx.fillText(`M${idx + 1}`, 78 + idx * 82, 18);
  });
}

function formatNumber(value) {
  const number = Number(value || 0);
  if (!Number.isFinite(number)) return "--";
  if (Math.abs(number) >= 100) return number.toFixed(1);
  return number.toFixed(3).replace(/0+$/, "").replace(/\.$/, "");
}

function updatePendingCount() {
  $("pendingCount").textContent = pending.run_status.size + pending.set_values.size;
  $("commandState").textContent = pending.run_status.size + pending.set_values.size ? "待发送" : "待命";
}

document.addEventListener("change", (event) => {
  const runKey = event.target.dataset.runKey;
  if (runKey) {
    const [dev_type, dev_name] = runKey.split("|");
    pending.run_status.set(runKey, { dev_type, dev_name, run_stat: event.target.checked ? 1 : 0 });
    updatePendingCount();
  }
});

document.addEventListener("input", (event) => {
  const setKey = event.target.dataset.setKey;
  if (setKey) {
    const [dev_type, dev_name] = setKey.split("|");
    const set_type = event.target.dataset.setType;
    pending.set_values.set(`${setKey}|${set_type}`, {
      dev_type,
      dev_name,
      set_type,
      set_value: Number(event.target.value),
    });
    updatePendingCount();
  }
});

$("sendCommands").addEventListener("click", async () => {
  const body = {
    source: "trainee-ui",
    run_status: Array.from(pending.run_status.values()),
    set_values: Array.from(pending.set_values.values()),
  };
  await api("/api/student/commands", { method: "POST", body: JSON.stringify(body) });
  pending.run_status.clear();
  pending.set_values.clear();
  updatePendingCount();
  await refresh();
});

setInterval(refresh, 2000);
refresh();
