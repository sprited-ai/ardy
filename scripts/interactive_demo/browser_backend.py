# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the diffusion window in the viewer's browser (onnxruntime-web on WebGPU) instead of the server model.

The demo keeps its viser UI, text encoder, playback and constraints on the server. A "worker" page
(``web/hybrid.html``, which also embeds the viser UI, or the bare ``web/worker.html``) loads the
single-window ONNX graph exported by ``web/export_web_onnx.py`` and connects back over a WebSocket.
For every text-only window the server sends ``history / text embedding / noise / guidance weights``
and gets the normalized motion features back; windows that the fixed graph cannot express (first
window without history, kinematic constraints, a different denoising-step count or history length)
fall back to the PyTorch model transparently.
"""

import asyncio
import base64
import functools
import http.server
import json
import os
import threading
import time
from typing import Optional

import numpy as np
import torch

WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "web")


class _StaticHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def log_message(self, *args):  # keep the demo console readable
        pass


class BrowserComputeServer:
    """WebSocket job server for browser workers plus a static server for the worker page and models."""

    def __init__(self, ws_port: int = 2334, http_port: int = 2335, web_dir: str = WEB_DIR):
        from websockets.asyncio.server import serve  # viser dependency

        self._serve = serve
        self.ws_port, self.http_port, self.web_dir = ws_port, http_port, web_dir
        self.loop = asyncio.new_event_loop()
        self.workers: dict = {}
        self.pending: dict = {}
        self._next_id = 0
        self.stats = {"jobs": 0, "last_ms": None, "last_roundtrip_ms": None, "failures": 0}
        threading.Thread(target=self._run_loop, daemon=True, name="browser-compute-ws").start()
        httpd = http.server.ThreadingHTTPServer(("0.0.0.0", http_port), functools.partial(_StaticHandler, directory=web_dir))
        threading.Thread(target=httpd.serve_forever, daemon=True, name="browser-compute-http").start()
        print(f"[browser backend] open http://127.0.0.1:{http_port}/hybrid.html (viser UI + WebGPU worker); jobs on ws://127.0.0.1:{ws_port}")

    # ------------------------------------------------------------------ asyncio side
    def _run_loop(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._main())

    async def _main(self) -> None:
        async with self._serve(self._handler, "0.0.0.0", self.ws_port, max_size=64 * 1024 * 1024):
            await asyncio.Future()

    async def _handler(self, ws) -> None:
        self.workers[ws] = {"backend": "?", "agent": "", "connected_at": time.time()}
        try:
            async for raw in ws:
                msg = json.loads(raw)
                kind = msg.get("type")
                if kind == "hello":
                    self.workers[ws].update(backend=msg.get("backend", "?"), agent=msg.get("agent", ""))
                    print(f"[browser backend] worker connected ({msg.get('backend')}), {len(self.workers)} total")
                elif kind in ("result", "error"):
                    fut = self.pending.pop(msg.get("id"), None)
                    if fut is None or fut.done():
                        continue
                    if kind == "result":
                        fut.set_result(msg)
                    else:
                        fut.set_exception(RuntimeError(msg.get("message", "worker error")))
        finally:
            self.workers.pop(ws, None)
            print(f"[browser backend] worker disconnected, {len(self.workers)} left")

    async def _submit(self, payload: dict):
        ws = next(iter(self.workers), None)
        if ws is None:
            raise RuntimeError("no browser worker connected")
        self._next_id += 1
        payload["id"] = self._next_id
        fut = self.loop.create_future()
        self.pending[payload["id"]] = fut
        await ws.send(json.dumps(payload))
        return await fut

    # ------------------------------------------------------------------ thread side
    @property
    def connected(self) -> bool:
        return bool(self.workers)

    def worker_backend(self) -> Optional[str]:
        for info in self.workers.values():
            return info.get("backend")
        return None

    def run_window(self, history, text_feat, noise, cfg_text: float, cfg_cstr: float, timeout: float = 60.0) -> np.ndarray:
        """Blocking call from the generation thread. Returns the normalized motion [1, T, D] (float32)."""
        b64 = lambda a: base64.b64encode(np.ascontiguousarray(a, dtype=np.float32).tobytes()).decode()  # noqa: E731
        payload = {
            "type": "job", "history": b64(history), "history_shape": list(history.shape), "text_feat": b64(text_feat),
            "noise": b64(noise), "noise_shape": list(noise.shape), "cfg_text": float(cfg_text), "cfg_cstr": float(cfg_cstr),
        }
        started = time.perf_counter()
        try:
            msg = asyncio.run_coroutine_threadsafe(self._submit(payload), self.loop).result(timeout=timeout)
        except Exception:
            self.stats["failures"] += 1
            raise
        self.stats["jobs"] += 1
        self.stats["last_ms"] = msg.get("ms")
        self.stats["last_roundtrip_ms"] = round((time.perf_counter() - started) * 1000)
        return np.frombuffer(base64.b64decode(msg["motion"]), dtype=np.float32).reshape(msg["shape"]).copy()

    def status_line(self) -> str:
        if not self.connected:
            return f"browser worker: not connected, open http://127.0.0.1:{self.http_port}/hybrid.html"
        s = self.stats
        line = f"browser worker: connected ({self.worker_backend()}), windows {s['jobs']}"
        if s["last_ms"] is not None:
            line += f", last {s['last_ms']} ms compute / {s['last_roundtrip_ms']} ms round trip"
        if s["failures"]:
            line += f", {s['failures']} failed"
        return line


class BrowserBackend:
    """Stands in for ``Ardy.autoregressive_step`` on text-only windows; everything else stays on the server model."""

    def __init__(self, server: BrowserComputeServer, window_cfg: dict, device):
        self.server, self.cfg, self.device = server, window_cfg, device

    def eligible(self, num_frames, num_denoising_steps, motion_mask, init_history_sequence, num_samples) -> tuple:
        H, G = self.cfg["history_frames"], self.cfg["gen_frames"]
        if not self.server.connected:
            return False, "no browser worker connected"
        if init_history_sequence is None:
            return False, "first window has no history"
        if init_history_sequence.shape[1] != H:
            return False, f"history length {init_history_sequence.shape[1]} (graph needs {H})"
        if num_frames != H + G:
            return False, f"window of {num_frames} frames (graph needs {H + G})"
        if int(num_denoising_steps) != int(self.cfg["num_denoising_steps"]):
            return False, f"{num_denoising_steps} denoising steps (graph has {self.cfg['num_denoising_steps']})"
        if num_samples != 1:
            return False, f"{num_samples} samples (graph runs one)"
        if motion_mask is not None and bool(motion_mask.any()):
            return False, "kinematic constraints in the window"
        return True, ""

    def autoregressive_step(self, num_frames, num_denoising_steps, motion_mask, observed_motion, cfg_weight=2.0,
                            texts=None, text_feat=None, text_pad_mask=None, init_history_sequence=None, **_ignored):
        w_text, w_cstr = cfg_weight if isinstance(cfg_weight, (tuple, list)) else (cfg_weight, 0.0)
        noise = torch.randn(1, self.cfg["gen_frames"] // 4, self.cfg["token_dim"])  # torch RNG: follows seed_everything
        history = init_history_sequence[:1].detach().float().cpu().numpy()
        feat = text_feat[:1, 0].detach().float().cpu().numpy()
        started = time.perf_counter()
        motion = self.server.run_window(history, feat, noise.numpy(), w_text, w_cstr)
        print(f"[browser backend] window in {(time.perf_counter() - started) * 1000:.0f} ms "
              f"({self.server.stats['last_ms']} ms in the browser)")
        return torch.from_numpy(motion).to(self.device)


class BrowserBackendMixin:
    """Demo glue: start the compute server, pick the backend per window, report status."""

    def init_browser_backend(self, default_backend: str = "server", ws_port: int = 2334, http_port: int = 2335) -> None:
        self.default_backend = default_backend
        self.browser_server = None
        self.browser_backend = None
        self._browser_fallback_reason = None
        cfg_path = os.path.join(WEB_DIR, "models", "window.json")
        onnx_path = os.path.join(WEB_DIR, "models", "window.onnx")
        if not (os.path.exists(cfg_path) and os.path.exists(onnx_path)):
            if default_backend == "browser":
                print(f"[browser backend] disabled: {onnx_path} missing; run `python web/export_web_onnx.py` first")
            return
        self.browser_server = BrowserComputeServer(ws_port=ws_port, http_port=http_port)
        self.browser_backend = BrowserBackend(self.browser_server, json.load(open(cfg_path)), self.device)

    def session_wants_browser(self, session) -> bool:
        handle = getattr(session.gui_elements, "gui_generation_backend", None)
        return handle is not None and handle.value.startswith("browser")

    def pick_generation_backend(self, session, **window):
        """The object whose ``autoregressive_step`` runs this window: the browser backend when it can, else the model."""
        if self.browser_backend is None or not self.session_wants_browser(session):
            return session.model
        ok, reason = self.browser_backend.eligible(**window)
        if ok:
            self._browser_fallback_reason = None
            return self.browser_backend
        if reason != self._browser_fallback_reason:
            print(f"[browser backend] falling back to the server model: {reason}")
            self._browser_fallback_reason = reason
        return session.model

    def browser_backend_status(self) -> Optional[str]:
        server = getattr(self, "browser_server", None)  # the status sampler may run before init_browser_backend
        return server.status_line() if server is not None else None
