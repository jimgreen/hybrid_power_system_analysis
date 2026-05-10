(() => {
  const tableHost = document.getElementById("usersTable");
  const status = document.getElementById("usersStatus");

  async function requestJson(url, options = {}) {
    const response = await fetch(url, {
      credentials: "same-origin",
      headers: { "Content-Type": "application/json", ...(options.headers || {}) },
      ...options,
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(data.message || "请求失败");
    }
    return data;
  }

  function escapeHtml(value) {
    return String(value ?? "").replace(/[&<>"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[char]);
  }

  function render(users) {
    if (!tableHost) return;
    tableHost.innerHTML = `<table><thead><tr><th>用户名称</th><th>权限</th><th>创建时间</th><th>操作</th></tr></thead><tbody>${users
      .map((user) => `<tr data-user-id="${user.id}">
        <td>${escapeHtml(user.username)}</td>
        <td><select data-role><option value="admin" ${user.role === "admin" ? "selected" : ""}>管理员</option><option value="user" ${user.role === "user" ? "selected" : ""}>普通用户</option></select></td>
        <td>${escapeHtml(user.created_at)}</td>
        <td><button class="danger" data-delete-user type="button">删除</button></td>
      </tr>`)
      .join("")}</tbody></table>`;
    tableHost.querySelectorAll("[data-role]").forEach((select) => {
      select.addEventListener("change", () => updateRole(select.closest("[data-user-id]").dataset.userId, select.value));
    });
    tableHost.querySelectorAll("[data-delete-user]").forEach((button) => {
      button.addEventListener("click", () => deleteUser(button.closest("[data-user-id]").dataset.userId));
    });
  }

  async function loadUsers() {
    const data = await requestJson("/api/users");
    render(data.users || []);
  }

  async function updateRole(id, role) {
    if (status) status.textContent = "正在保存...";
    try {
      await requestJson(`/api/users/${encodeURIComponent(id)}`, { method: "PUT", body: JSON.stringify({ role }) });
      if (status) status.textContent = "已保存";
      await loadUsers();
    } catch (error) {
      if (status) status.textContent = error.message;
      await loadUsers().catch(() => null);
    }
  }

  async function deleteUser(id) {
    if (!confirm("确定删除该用户？")) return;
    if (status) status.textContent = "正在删除...";
    try {
      await requestJson(`/api/users/${encodeURIComponent(id)}`, { method: "DELETE" });
      if (status) status.textContent = "已删除";
      await loadUsers();
    } catch (error) {
      if (status) status.textContent = error.message;
    }
  }

  loadUsers().catch((error) => {
    if (status) status.textContent = error.message;
  });
})();
