# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Text encoder that loads on demand and unloads itself after an idle period.

The demo's text encoder (LLM2Vec on Llama-3-8B: ~5 GB as int4 on MPS, 16 GB in bf16) is only needed
for a few hundred milliseconds per new prompt, yet it used to stay resident for the whole session.
:class:`IdleUnloadingTextEncoder` keeps the "build once, share everywhere" contract of
``run_demo.py`` but owns the encoder's lifetime:

- loaded at construction (``preload=True``) so the first prompt is fast,
- unloaded after ``idle_timeout`` seconds without an encode call, returning its memory to the
  device (MPS / CUDA),
- rebuilt transparently by the next encode call (~2 s for the int4 checkpoint).

Combined with :class:`~interactive_demo.embedding_cache.CachedTextEncoder` on top, a cached
prompt never triggers a reload at all.
"""

import gc
import threading
import time
from typing import Callable, Optional

import torch

from ardy.model.load_model import text_encoder_half_dtype
from ardy.tools import get_default_device

DEFAULT_IDLE_TIMEOUT_S = 300.0


def _is_local_encoder(encoder) -> bool:
    """True for in-process encoders (something to free); False for the remote API client."""
    return getattr(encoder, "model", None) is not None or callable(getattr(encoder, "parameters", None))


def _device_of(encoder) -> str:
    device = getattr(encoder, "device", None)
    if device is None:
        for source in (encoder, getattr(encoder, "model", None)):
            params = getattr(source, "parameters", None)
            if callable(params):
                try:
                    device = next(params()).device
                    break
                except (StopIteration, TypeError):
                    continue
    return str(device) if device is not None else "cpu"


def _dtype_of(encoder) -> Optional[torch.dtype]:
    dtype = getattr(encoder, "dtype", None)
    if dtype is None:
        for source in (encoder, getattr(encoder, "model", None)):
            params = getattr(source, "parameters", None)
            if callable(params):
                try:
                    return next(params()).dtype
                except (StopIteration, TypeError):
                    continue
    return dtype


def _accelerator() -> Optional[str]:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return None


def accelerator_memory_bytes() -> int:
    """Memory the driver currently holds for the accelerator (what unloading should give back)."""
    acc = _accelerator()
    if acc == "mps":
        return int(torch.mps.driver_allocated_memory())
    if acc == "cuda":
        return int(torch.cuda.memory_reserved())
    return 0


def release_accelerator_memory() -> None:
    """Return cached allocator blocks to the driver.

    Always targets the accelerator, not the encoder's last device: an encoder moved to CPU via
    ``.to("cpu")`` leaves its old blocks in the MPS/CUDA cache until someone empties it.
    """
    gc.collect()
    acc = _accelerator()
    if acc == "mps":
        torch.mps.synchronize()
        torch.mps.empty_cache()
    elif acc == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


class IdleUnloadingTextEncoder:
    """Wrap an encoder *factory* so the encoder exists only while it is being used.

    Presents the same surface as the encoders it wraps (``__call__(texts) -> (tensor, lengths)``,
    ``to(device, dtype)``, ``device`` / ``dtype`` attributes, attribute pass-through), so it can be
    dropped in wherever ``load_text_encoder()``'s result was used.

    Thread-safety: all state changes happen under one re-entrant lock. An encode call that is in
    flight (``_inflight > 0``) blocks unloading; the idle timer simply re-arms and checks again.
    """

    def __init__(
        self,
        factory: Callable[[], object],
        idle_timeout: float = DEFAULT_IDLE_TIMEOUT_S,
        preload: bool = True,
        name: str = "text encoder",
    ) -> None:
        self._factory = factory
        self.idle_timeout = float(idle_timeout)
        self._name = name
        self._lock = threading.RLock()
        self._encoder = None
        self._inflight = 0
        self._last_used = time.monotonic()
        self._timer: Optional[threading.Timer] = None
        # Placement requested through .to(); re-applied whenever the encoder is (re)built.
        self._device: Optional[object] = None
        self._dtype: Optional[torch.dtype] = None
        self._unloadable = True
        if preload:
            self._ensure_loaded()

    # ------------------------------------------------------------------ state
    @property
    def loaded(self) -> bool:
        with self._lock:
            return self._encoder is not None

    @property
    def device(self):
        with self._lock:
            encoder = self._encoder
        if encoder is not None:
            return _device_of(encoder)
        return self._device if self._device is not None else get_default_device()

    @property
    def dtype(self):
        with self._lock:
            encoder = self._encoder
        if encoder is not None:
            dtype = _dtype_of(encoder)
            if dtype is not None:
                return dtype
        return self._dtype if self._dtype is not None else text_encoder_half_dtype(self.device)

    # --------------------------------------------------------------- lifecycle
    def _ensure_loaded(self):
        with self._lock:
            if self._encoder is None:
                started = time.monotonic()
                encoder = self._factory()
                if (self._device is not None or self._dtype is not None) and hasattr(encoder, "to"):
                    encoder.to(device=self._device, dtype=self._dtype)
                self._encoder = encoder
                self._unloadable = _is_local_encoder(encoder)
                note = "" if self._unloadable else " (remote service; idle unload disabled)"
                print(f"[{self._name}] loaded on {_device_of(encoder)} in {time.monotonic() - started:.1f} s{note}")
            self._touch()
            return self._encoder

    def _touch(self) -> None:
        """Record use and (re)start the idle countdown. Caller holds the lock."""
        self._last_used = time.monotonic()
        self._arm_timer(self.idle_timeout)

    def _arm_timer(self, delay: float) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        if self.idle_timeout <= 0 or not self._unloadable or self._encoder is None:
            return
        timer = threading.Timer(max(0.0, delay), self._on_timer)
        timer.daemon = True
        timer.start()
        self._timer = timer

    def _on_timer(self) -> None:
        with self._lock:
            self._timer = None
            if self._encoder is None:
                return
            idle = time.monotonic() - self._last_used
            if self._inflight > 0:
                self._arm_timer(self.idle_timeout)  # an encode is running; look again later
                return
            if idle < self.idle_timeout:
                self._arm_timer(self.idle_timeout - idle)  # used after the timer was armed
                return
            self.unload(reason=f"idle for {idle:.0f} s")

    def unload(self, reason: str = "requested") -> None:
        """Drop the encoder and return its memory to the device. No-op when not loaded."""
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            encoder = self._encoder
            if encoder is None:
                return
            self._encoder = None
            before = accelerator_memory_bytes()
            del encoder
            release_accelerator_memory()
            after = accelerator_memory_bytes()
        freed = f"; {_accelerator()} memory {before / 1e9:.1f} -> {after / 1e9:.1f} GB" if before else ""
        print(f"[{self._name}] unloaded ({reason}){freed}; it is rebuilt on the next new prompt")

    # ------------------------------------------------------------ encoder API
    def __call__(self, texts):
        with self._lock:
            encoder = self._ensure_loaded()
            self._inflight += 1
        try:
            return encoder(texts)
        finally:
            with self._lock:
                self._inflight -= 1
                self._touch()

    def to(self, device=None, dtype=None):
        with self._lock:
            if device is not None:
                self._device = device
            if dtype is not None:
                self._dtype = dtype
            if self._encoder is not None and hasattr(self._encoder, "to"):
                self._encoder.to(device=device, dtype=dtype)
        return self

    def __getattr__(self, name):
        # Only reached for names not defined on the wrapper. Never triggers a load: a plain
        # attribute lookup must not cost a multi-GB model build.
        if name.startswith("_"):
            raise AttributeError(name)
        with self._lock:
            encoder = self._encoder
        if encoder is None:
            raise AttributeError(f"{name!r}: {self._name} is not loaded right now")
        return getattr(encoder, name)
