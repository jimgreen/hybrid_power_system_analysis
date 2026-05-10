(() => {
  const loginUrl = () => `/login.html?next=${encodeURIComponent(location.pathname + location.search + location.hash)}`;

  async function requestJson(url, options = {}) {
    const response = await fetch(url, {
      credentials: "same-origin",
      headers: { "Content-Type": "application/json", ...(options.headers || {}) },
      ...options,
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      const error = new Error(data.message || "请求失败");
      error.status = response.status;
      throw error;
    }
    return data;
  }

  function renderUser(user) {
    document.querySelectorAll("[data-auth-username]").forEach((target) => {
      target.textContent = user.username;
    });
    document.querySelectorAll("[data-auth-role]").forEach((target) => {
      target.textContent = user.role === "admin" ? "管理员" : "普通用户";
    });
    document.querySelectorAll("[data-auth-user]").forEach((target) => {
      target.hidden = false;
    });
    document.querySelectorAll("[data-admin-only]").forEach((target) => {
      target.hidden = user.role !== "admin";
    });
  }

  async function logout() {
    await requestJson("/api/auth/logout", { method: "POST", body: "{}" }).catch(() => null);
    location.href = "/login.html";
  }

  async function boot() {
    try {
      const data = await requestJson("/api/auth/me");
      renderUser(data.user);
      if (document.body.dataset.adminPage === "true" && data.user.role !== "admin") {
        location.replace("/index.html");
        return;
      }
    } catch (error) {
      location.replace(loginUrl());
      return;
    }
    document.querySelectorAll("[data-logout]").forEach((button) => {
      button.addEventListener("click", logout);
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
