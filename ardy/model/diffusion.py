# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Diffusion process and DDIM sampling for motion generation."""

import math
from typing import Optional, Tuple

import torch
from torch import nn


def get_beta_schedule(
    num_diffusion_timesteps: int,
    max_beta: Optional[float] = 0.999,
) -> torch.Tensor:
    """Get cosine beta schedule."""

    def alpha_bar(t):
        return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float)


class Diffusion(torch.nn.Module):
    """Cosine-schedule diffusion process: betas, alphas, and DDIM step mapping."""

    def __init__(self, num_base_steps: int):
        """Set up cosine beta schedule and precompute diffusion variables for num_base_steps."""
        super().__init__()
        self.num_base_steps = num_base_steps
        betas_base = get_beta_schedule(self.num_base_steps)
        self.register_buffer("betas_base", betas_base, persistent=False)
        alphas_cumprod_base = torch.cumprod(1.0 - self.betas_base, dim=0)
        self.register_buffer("alphas_cumprod_base", alphas_cumprod_base, persistent=False)
        # Active schedule cache (see ensure_schedule). map_tensor is a (non-persistent) buffer so
        # it follows the module across .to(device).
        self._sched_n: Optional[int] = None
        self._sched_src = None
        self.register_buffer("map_tensor", torch.zeros(0, dtype=torch.long), persistent=False)
        self.ensure_schedule(self.num_base_steps)

    def extra_repr(self) -> str:
        return f"num_base_steps={self.num_base_steps}"

    @property
    def device(self):
        return self.betas_base.device

    def space_timesteps(self, num_denoising_steps: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (use_timesteps, map_tensor) for a subsampled denoising schedule of
        num_denoising_steps."""
        nsteps_train = self.num_base_steps
        # Plain-int arithmetic: a tensor here would force a host sync at every call.
        num_denoising_steps = int(num_denoising_steps)
        frac_stride = (nsteps_train - 1) / max(1, num_denoising_steps - 1)
        use_timesteps = torch.round(torch.arange(nsteps_train, device=self.device) * frac_stride).to(torch.long)
        use_timesteps = torch.clamp(use_timesteps, max=nsteps_train - 1)
        map_tensor = torch.arange(nsteps_train, device=self.device, dtype=torch.long)[use_timesteps]
        return use_timesteps, map_tensor

    def ensure_schedule(self, num_denoising_steps) -> torch.Tensor:
        """Make the diffusion buffers current for ``num_denoising_steps`` and return ``map_tensor``.

        ``num_denoising_steps`` may be a Python int or a (1-element) tensor. The schedule is only
        recomputed when the step count changes; the same tensor object passed again (as the
        denoising loop does every step) is recognised without reading it back from the device.
        """
        if isinstance(num_denoising_steps, torch.Tensor):
            if num_denoising_steps is self._sched_src and self._sched_n is not None:
                return self.map_tensor
            n = int(num_denoising_steps.reshape(-1)[0])
        else:
            n = int(num_denoising_steps)
        if n != self._sched_n or self.map_tensor.device != self.device:
            use_timesteps, map_tensor = self.space_timesteps(n)
            self.calc_diffusion_vars(use_timesteps)
            self._sched_n = n
            self.map_tensor = map_tensor  # buffer assignment
        self._sched_src = num_denoising_steps if isinstance(num_denoising_steps, torch.Tensor) else None
        return self.map_tensor

    def calc_diffusion_vars(self, use_timesteps: torch.Tensor) -> None:
        """Update buffers (betas, alphas, alphas_cumprod, etc.) for the given subsampled
        timesteps."""
        alphas_cumprod = self.alphas_cumprod_base[use_timesteps]
        last_alpha_cumprod = torch.cat([alphas_cumprod.new_ones(1), alphas_cumprod[:-1]])
        betas = 1.0 - alphas_cumprod / last_alpha_cumprod
        self.register_buffer("betas", betas, persistent=False)

        alphas = 1.0 - self.betas
        self.register_buffer("alphas", alphas, persistent=False)
        alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        alphas_cumprod = torch.clamp(alphas_cumprod, min=1e-9)
        self.register_buffer("alphas_cumprod", alphas_cumprod, persistent=False)

        alphas_cumprod_prev = torch.cat([self.alphas_cumprod.new_ones(1), self.alphas_cumprod[:-1]])
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev, persistent=False)

        sqrt_recip_alphas_cumprod = torch.rsqrt(self.alphas_cumprod)
        self.register_buffer("sqrt_recip_alphas_cumprod", sqrt_recip_alphas_cumprod, persistent=False)

        sqrt_recipm1_alphas_cumprod = torch.rsqrt(self.alphas_cumprod / (1.0 - self.alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", sqrt_recipm1_alphas_cumprod, persistent=False)

        posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance, persistent=False)

        sqrt_alphas_cumprod = torch.rsqrt(1.0 / self.alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", sqrt_alphas_cumprod, persistent=False)

        sqrt_one_minus_alphas_cumprod = torch.rsqrt(1.0 / (1.0 - self.alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            sqrt_one_minus_alphas_cumprod,
            persistent=False,
        )

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor = None,
    ):
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape

        xt = (
            self.sqrt_alphas_cumprod[t, None, None] * x_start
            + self.sqrt_one_minus_alphas_cumprod[t, None, None] * noise
        )
        return xt


class DDIMSampler(nn.Module):
    """Deterministic DDIM sampler (eta = 0)."""

    def __init__(self, diffusion: Diffusion):
        super().__init__()
        self.diffusion = diffusion

    def __call__(
        self,
        x_t: torch.Tensor,
        pred_xstart: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        # Assumes the diffusion buffers are current for the active schedule
        # (denoising_step refreshes them every step before calling).
        eps = (
            self.diffusion.sqrt_recip_alphas_cumprod[t, None, None] * x_t - pred_xstart
        ) / self.diffusion.sqrt_recipm1_alphas_cumprod[t, None, None]
        alpha_bar_prev = self.diffusion.alphas_cumprod_prev[t, None, None]
        x = pred_xstart * torch.sqrt(alpha_bar_prev) + torch.sqrt(1 - alpha_bar_prev) * eps
        return x
