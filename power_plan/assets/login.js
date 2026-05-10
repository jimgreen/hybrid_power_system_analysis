(() => {
  const params = new URLSearchParams(location.search);
  const nextUrl = params.get("next") || "/index.html";

  async function requestJson(url, body) {
    const response = await fetch(url, {
      method: "POST",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(data.message || "请求失败");
    }
    return data;
  }

  function bindForm(formId, endpoint) {
    const form = document.getElementById(formId);
    if (!form) return;
    const status = form.querySelector("[data-auth-status]");
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const payload = Object.fromEntries(new FormData(form).entries());
      if (status) status.textContent = "处理中...";
      try {
        await requestJson(endpoint, payload);
        location.href = nextUrl;
      } catch (error) {
        if (status) status.textContent = error.message;
      }
    });
  }

  bindForm("loginForm", "/api/auth/login");
  bindForm("registerForm", "/api/auth/register");
})();
