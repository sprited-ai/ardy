# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Part of InteractiveTimelineDemo: the Debug panel (memory + text-encoder state), the progress feedback
while a prompt is being encoded, and the "Now playing" prompt that follows playback."""

import resource
import sys
import threading
import time

import torch

from .common import *  # noqa: F401,F403

try:
    import psutil
except ImportError:  # pragma: no cover - optional
    psutil = None

_TLS = threading.local()  # the notification/client of the prompt encode running on this thread


def memory_snapshot() -> tuple:
    """(rss_bytes, peak_rss_bytes, accelerator dict) for this process."""
    rss = psutil.Process().memory_info().rss if psutil is not None else 0
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    accel = {}
    if torch.cuda.is_available():
        accel = {"name": "cuda", "current": torch.cuda.memory_allocated(), "driver": torch.cuda.memory_reserved(),
                 "peak": torch.cuda.max_memory_allocated()}
    elif torch.backends.mps.is_available():
        accel = {"name": "mps", "current": torch.mps.current_allocated_memory(), "driver": torch.mps.driver_allocated_memory()}
    return rss, peak, accel


class StatusMixin:
    # ------------------------------------------------------------ lifecycle
    def init_status(self) -> None:
        """Call once after ``self.text_encoder`` exists: hooks the encoder's events and starts the sampler."""
        self._accel_peak = 0
        lazy = getattr(self.text_encoder, "encoder", None)  # IdleUnloadingTextEncoder under the cache wrapper
        self._lazy_encoder = lazy if hasattr(lazy, "add_listener") else None
        if self._lazy_encoder is not None:
            self._lazy_encoder.add_listener(self._on_encoder_event)
        threading.Thread(target=self._status_sampler, daemon=True, name="status-sampler").start()

    def _on_encoder_event(self, event: str, info: dict) -> None:
        self._sample_peak()
        notif, client = getattr(_TLS, "notif", None), getattr(_TLS, "client", None)
        body = {
            "load_start": "Loading the text encoder (it was unloaded while idle)...",
            "load_end": f"Encoder loaded in {info.get('seconds', 0):.1f} s, encoding the prompt...",
            "encode_start": "Encoding the prompt...",
        }.get(event)
        if notif is not None and body:
            notif.body = body
            if client is not None:
                client.flush()

    # -------------------------------------------------------------- sampling
    def _sample_peak(self) -> tuple:
        rss, peak, accel = memory_snapshot()
        self._accel_peak = max(getattr(self, "_accel_peak", 0), accel.get("driver", 0), accel.get("peak", 0))
        return rss, peak, accel

    def _status_text(self) -> tuple:
        rss, peak, accel = self._sample_peak()
        mem = f"process RSS {rss / 1e9:.2f} GB, peak {peak / 1e9:.2f} GB"
        if accel:
            mem += (f"\n{accel['name']}: {accel['current'] / 1e9:.2f} GB live, "
                    f"{accel['driver'] / 1e9:.2f} GB held by driver, peak {self._accel_peak / 1e9:.2f} GB")
        browser = self.browser_backend_status() if hasattr(self, "browser_backend_status") else None
        if browser:
            mem += "\n" + browser
        enc = "remote service or none (nothing to unload)"
        if self._lazy_encoder is not None:
            s = self._lazy_encoder.status()
            if s["loaded"]:
                enc = f"loaded on {s['device']} ({s['dtype'].replace('torch.', '')}), idle {s['idle_seconds']:.0f} s"
                enc += f", unloads in {s['unload_in']:.0f} s" if s["unload_in"] is not None else ", kept resident"
            else:
                enc = "unloaded after idle, reloads on the next uncached prompt"
            if s["last_load_s"] is not None:
                enc += f"\nlast load {s['last_load_s']:.1f} s"
                if s["last_encode_s"] is not None:
                    enc += f", last encode {s['last_encode_s'] * 1000:.0f} ms"
        return mem, enc

    def _status_sampler(self) -> None:
        last = None
        while True:
            try:
                mem, enc = self._status_text()
                if (mem, enc) != last:
                    for session in list(getattr(self, "client_sessions", {}).values()):
                        g = session.gui_elements
                        if getattr(g, "gui_debug_memory", None) is not None:
                            g.gui_debug_memory.value = mem
                            g.gui_debug_encoder.value = enc
                    last = (mem, enc)
            except Exception as e:
                print(f"[status] sampler error: {e!r}")
            time.sleep(1.0)

    # ------------------------------------------------------------------- GUI
    def _build_status_gui(self, client, g) -> None:
        with client.gui.add_folder("Debug", expand_by_default=False):
            g.gui_debug_memory = client.gui.add_text("Memory", "", multiline=True, disabled=True)
            g.gui_debug_encoder = client.gui.add_text("Text encoder", "", multiline=True, disabled=True)

    # ------------------------------------------------------- prompt encoding
    def encode_prompt(self, client_id: int, text: str, notify: bool = True) -> torch.Tensor:
        """Encode ``text`` with the shared encoder, showing what is happening: a loading toast whose body
        follows the encoder's events (reload after idle / encoding) and an animated bar under the prompt."""
        session = self.client_sessions[client_id]
        client, g = session.client, session.gui_elements
        bar = getattr(g, "gui_text_encoder_progress", None)
        notif = None
        if notify:
            notif = client.add_notification(title="Text encoder", body="Encoding the prompt...", loading=True, with_close_button=False)
            client.flush()
        if bar is not None:
            bar.visible = True
        _TLS.notif, _TLS.client = notif, client
        started = time.perf_counter()
        try:
            feat, _ = self.text_encoder([text])
        except Exception:
            if notif is not None:
                notif.title, notif.body, notif.color = "Text encoder failed", "See the console for details.", "red"
                notif.loading, notif.with_close_button = False, True
            raise
        finally:
            _TLS.notif = _TLS.client = None
            if bar is not None:
                bar.visible = False
        elapsed = time.perf_counter() - started
        if notif is not None:
            notif.title = "Prompt ready"
            notif.body = f"{text[:70]} ({elapsed * 1000:.0f} ms{', cached' if elapsed < 0.05 else ''})"
            notif.color, notif.loading, notif.with_close_button, notif.auto_close_seconds = "green", False, True, 2.5
        return feat.to(self.device)

    # ------------------------------------------------------------ now playing
    def note_prompt_start(self, session, start_frame: int, text: str) -> None:
        """Record that ``text`` drives the motion from ``start_frame`` on (playback reports it when it gets there)."""
        session.prompt_schedule = [(f, t) for f, t in session.prompt_schedule if f < start_frame] + [(start_frame, text)]

    def update_now_playing(self, client_id: int, frame_idx: int) -> None:
        session = self.client_sessions[client_id]
        text = None
        for start, t in session.prompt_schedule:
            if frame_idx >= start:
                text = t
        if text is None or text == session.now_playing_text:
            return
        session.now_playing_text = text
        g = session.gui_elements
        if getattr(g, "gui_now_playing", None) is not None:
            g.gui_now_playing.content = f"**Now playing:** {text}"
        session.client.add_notification(title="Now playing", body=f"{text} (from frame {frame_idx})", auto_close_seconds=3.0, color="teal")
