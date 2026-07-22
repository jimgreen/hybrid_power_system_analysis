"""HTTP server for the polar microgrid simulator and trainee consoles."""

from __future__ import annotations

import argparse
import json
import mimetypes
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urljoin, urlparse
from urllib.request import Request, urlopen

from hybrid_power_system_analysis.polar_microgrid_sim.service import MultiModelSimulator, PolarMicrogridSimulator
from hybrid_power_system_analysis.simu import simu_loop


PACKAGE_DIR = Path(__file__).resolve().parent
WEB_DIR = PACKAGE_DIR / "web"
CLOCK_BASE_INTERVAL_SECONDS = 1.0


def _clock_int_value(value: Any, default: int = 1) -> int:
    try:
        return max(1, int(round(float(value))))
    except (TypeError, ValueError):
        return default


class JsonApiError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def make_http_server(
    server_address: tuple[str, int],
    service: PolarMicrogridSimulator | MultiModelSimulator,
    *,
    role: str = "simulator",
    static_root: Optional[str | Path] = None,
    sim_url: Optional[str] = None,
) -> ThreadingHTTPServer:
    role = role.lower()
    if static_root is None:
        static_root = WEB_DIR / ("trainee" if role == "trainee" else "simulator")
    static_root = Path(static_root).resolve()
    sim_url = sim_url.rstrip("/") if sim_url else None

    class PolarMicrogridHandler(BaseHTTPRequestHandler):
        server_version = "PolarMicrogridHTTP/0.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def do_OPTIONS(self) -> None:
            self.send_response(204)
            self._cors()
            self.end_headers()

        def do_GET(self) -> None:
            try:
                if self.path.startswith("/api/") and role == "trainee" and sim_url:
                    self._proxy_to_simulator("GET", sim_url)
                    return
                if self.path.startswith("/api/"):
                    self._handle_api_get()
                    return
                self._serve_static(static_root)
            except JsonApiError as exc:
                self._send_json({"error": exc.message}, status=exc.status)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)

        def do_POST(self) -> None:
            try:
                if self.path.startswith("/api/") and role == "trainee" and sim_url:
                    self._proxy_to_simulator("POST", sim_url)
                    return
                self._handle_api_post()
            except JsonApiError as exc:
                self._send_json({"error": exc.message}, status=exc.status)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)

        def do_PUT(self) -> None:
            self.do_POST()

        def _request_model_id(self, payload: Optional[Mapping[str, Any]] = None) -> Optional[str]:
            parsed = urlparse(self.path)
            query_values = parse_qs(parsed.query).get("model_id") or parse_qs(parsed.query).get("model")
            if query_values and query_values[0]:
                return query_values[0]
            if payload:
                value = payload.get("model_id", payload.get("model"))
                return str(value) if value not in (None, "") else None
            return None

        def _target_service(self, payload: Optional[Mapping[str, Any]] = None) -> PolarMicrogridSimulator:
            if hasattr(service, "service_for"):
                try:
                    return service.service_for(self._request_model_id(payload))  # type: ignore[union-attr]
                except KeyError as exc:
                    raise JsonApiError(404, str(exc)) from exc
            return service  # type: ignore[return-value]

        def _model_catalog(self) -> Mapping[str, Any]:
            if hasattr(service, "models"):
                return {
                    "models": service.models(),  # type: ignore[union-attr]
                    "active_model_id": service.default_model_id,  # type: ignore[union-attr]
                }
            return {
                "models": [service.model_info()],  # type: ignore[union-attr]
                "active_model_id": service.model_id,  # type: ignore[union-attr]
            }

        def _handle_api_get(self) -> None:
            path = urlparse(self.path).path
            target = self._target_service()
            if path == "/api/health":
                self._send_json({"ok": True, "role": role})
            elif path == "/api/models":
                self._send_json(self._model_catalog())
            elif path == "/api/snapshot":
                self._send_json(target.snapshot())
            elif path == "/api/measurements":
                self._send_json(target.measurements())
            elif path == "/api/devices":
                self._send_json({"devices": target.devices()})
            elif path == "/api/curves":
                self._send_json(target.curves)
            elif path == "/api/settings":
                self._send_json(target.local_settings)
            elif path == "/api/config":
                self._send_json({"role": role, "sim_url": sim_url, "poll_ms": 2000, **self._model_catalog()})
            else:
                raise JsonApiError(404, f"Unknown API route: {path}")

        def _handle_api_post(self) -> None:
            path = urlparse(self.path).path
            payload = self._read_json_body()
            if path == "/api/models/clone":
                if not hasattr(service, "clone_model"):
                    raise JsonApiError(400, "Current simulator does not support multiple model folders")
                model_name = payload.get("name", payload.get("model_name", payload.get("new_model_id", "")))
                if not str(model_name or "").strip():
                    raise JsonApiError(400, "New model name is required")
                try:
                    model = service.clone_model(self._request_model_id(payload), model_name)  # type: ignore[union-attr]
                except ValueError as exc:
                    raise JsonApiError(400, str(exc)) from exc
                catalog = dict(self._model_catalog())
                catalog["active_model_id"] = model["id"]
                self._send_json({"model": model, **catalog})
                return

            target = self._target_service(payload)
            if path == "/api/student/commands":
                self._send_json(target.apply_student_commands(payload, source=str(payload.get("source", "student"))))
            elif path == "/api/clock":
                self._send_json(target.control_clock(payload))
            elif path == "/api/step":
                self._send_json(target.step())
            elif path == "/api/curves":
                self._send_json(target.set_curves(payload))
            elif path == "/api/settings":
                self._send_json(target.set_local_settings(payload))
            elif path == "/api/device-faults":
                self._send_json(target.set_local_settings({"device_faults": payload.get("items", payload)}))
            elif path == "/api/measurement-faults":
                self._send_json(target.set_local_settings({"measurement_faults": payload.get("items", payload)}))
            elif path == "/api/modes":
                self._send_json(target.set_local_settings({"modes": payload.get("items", payload)}))
            else:
                raise JsonApiError(404, f"Unknown API route: {path}")

        def _read_json_body(self) -> Mapping[str, Any]:
            length = int(self.headers.get("Content-Length", "0") or 0)
            if length <= 0:
                return {}
            raw = self.rfile.read(length)
            if not raw:
                return {}
            try:
                payload = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError as exc:
                raise JsonApiError(400, f"Invalid JSON payload: {exc}") from exc
            if not isinstance(payload, Mapping):
                raise JsonApiError(400, "JSON payload must be an object")
            return payload

        def _serve_static(self, root: Path) -> None:
            path = urlparse(self.path).path
            rel = "index.html" if path in ("", "/") else path.lstrip("/")
            target = (root / rel).resolve()
            if not str(target).startswith(str(root)) or not target.exists() or not target.is_file():
                target = root / "index.html"
            if not target.exists():
                raise JsonApiError(404, f"Static file not found: {rel}")
            content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
            data = target.read_bytes()
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _proxy_to_simulator(self, method: str, base_url: str) -> None:
            path = self.path
            body = None
            headers = {"Accept": "application/json"}
            if method in ("POST", "PUT"):
                length = int(self.headers.get("Content-Length", "0") or 0)
                body = self.rfile.read(length) if length else b"{}"
                headers["Content-Type"] = self.headers.get("Content-Type", "application/json")
            request = Request(urljoin(base_url + "/", path.lstrip("/")), data=body, headers=headers, method=method)
            try:
                with urlopen(request, timeout=10) as response:
                    data = response.read()
                    status = response.status
                    content_type = response.headers.get("Content-Type", "application/json")
            except HTTPError as exc:
                data = exc.read()
                status = exc.code
                content_type = exc.headers.get("Content-Type", "application/json")
            except URLError as exc:
                raise JsonApiError(502, f"Simulator is unreachable: {exc}") from exc
            self.send_response(status)
            self._cors()
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_json(self, payload: Any, status: int = 200) -> None:
            data = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
            self.send_response(status)
            self._cors()
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _cors(self) -> None:
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
            self.send_header("Cache-Control", "no-store")

    server = ThreadingHTTPServer(server_address, PolarMicrogridHandler)
    server.service = service  # type: ignore[attr-defined]
    return server


def _advance_clock_if_due(service: PolarMicrogridSimulator, last_step: float) -> float:
    clock = service.snapshot()["clock"]
    now = time.monotonic()
    if clock["state"] != "running":
        return now
    if now - last_step < CLOCK_BASE_INTERVAL_SECONDS:
        return last_step
    speed_minutes = _clock_int_value(clock.get("speed"), 1)
    step_minutes = _clock_int_value(clock.get("step_minutes"), 1)
    # Do not catch up by elapsed wall time: one completed solve advances one logical simulation step.
    advance_minutes = speed_minutes * step_minutes
    try:
        service.step(advance_minutes=advance_minutes)
    except Exception:
        service.control_clock({"action": "pause"})
    return time.monotonic()


def start_clock_worker(service: PolarMicrogridSimulator, stop_event: threading.Event) -> threading.Thread:
    def worker() -> None:
        last_step = time.monotonic()
        while not stop_event.is_set():
            last_step = _advance_clock_if_due(service, last_step)
            stop_event.wait(0.05)

    thread = threading.Thread(target=worker, name=f"polar-microgrid-clock-{service.model_id}", daemon=True)
    thread.start()
    return thread


def start_multi_model_clock_worker(service: MultiModelSimulator, stop_event: threading.Event) -> threading.Thread:
    def worker() -> None:
        last_steps: dict[str, float] = {}
        while not stop_event.is_set():
            current_ids = set()
            for item in service.iter_services():
                current_ids.add(item.model_id)
                last_steps[item.model_id] = _advance_clock_if_due(
                    item,
                    last_steps.get(item.model_id, time.monotonic()),
                )
            for stale_id in set(last_steps) - current_ids:
                last_steps.pop(stale_id, None)
            stop_event.wait(0.05)

    thread = threading.Thread(target=worker, name="polar-microgrid-clock-models", daemon=True)
    thread.start()
    return thread


def start_clock_workers(
    service: PolarMicrogridSimulator | MultiModelSimulator,
    stop_event: threading.Event,
) -> list[threading.Thread]:
    if hasattr(service, "iter_services"):
        return [start_multi_model_clock_worker(service, stop_event)]  # type: ignore[arg-type]
    return [start_clock_worker(service, stop_event)]  # type: ignore[arg-type]


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve polar microgrid simulator or trainee console.")
    parser.add_argument("--role", choices=("simulator", "trainee"), default="simulator")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--sim-dir", default=str(simu_loop.SIMU_DIR))
    parser.add_argument("--models-dir", default=None, help="Directory whose direct subfolders are simulation models.")
    parser.add_argument("--runtime-dir", default=None)
    parser.add_argument("--sim-url", default=None, help="Simulator API base URL for trainee proxy mode.")
    parser.add_argument("--static-root", default=None)
    parser.add_argument("--no-worker", action="store_true", help="Do not start automatic clock worker.")
    parser.add_argument("--noise-std", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    port = args.port if args.port is not None else (8720 if args.role == "trainee" else 8710)
    sim_dir = Path(args.sim_dir).resolve()
    runtime_dir = Path(args.runtime_dir).resolve() if args.runtime_dir else sim_dir / f"runtime_{args.role}"
    service = MultiModelSimulator.discover(
        sim_dir=sim_dir,
        runtime_dir=runtime_dir,
        noise_std=args.noise_std,
        random_seed=args.seed,
        models_dir=args.models_dir,
    )
    server = make_http_server(
        (args.host, port),
        service,
        role=args.role,
        static_root=args.static_root,
        sim_url=args.sim_url,
    )
    stop_event = threading.Event()
    workers = [] if args.no_worker else start_clock_workers(service, stop_event)
    print(f"{args.role} console: http://{args.host}:{port}/")
    print(f"runtime dir: {runtime_dir}")
    print(f"models dir: {service.models_root}")
    print(f"models: {', '.join(item['id'] for item in service.models())}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        for worker in workers:
            worker.join(timeout=2)
        server.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
