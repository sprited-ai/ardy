# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One fixed-shape autoregressive window of ARDY as a single graph, for the browser (onnxruntime-web).

``WebWindow`` re-implements :meth:`ardy.model.ardy_model.Ardy.autoregressive_step` for the text-only
case with every shape fixed (history frames, generation horizon, denoising steps) so it traces to one
ONNX graph: history tokenization (autoencoder encoder), recentering, the unrolled CFG denoising loop
with the DDIM sampler, root translation back to world space, decoding, and the motion-representation
inverse (forward kinematics) that yields joint positions. The browser then only slices the last
``history_frames`` frames of each output to feed the next window.

Differences from ``autoregressive_step`` are deliberate and verified in ``export_web_window``:
the initial noise and the two guidance weights are graph inputs (deterministic, tunable at runtime)
instead of ``torch.randn`` / cached scalars.
"""

from typing import Tuple

import torch
from torch import nn

from ardy.model.ardy_model import Ardy, get_three_mask_from_len, translate_normalized_root_motion


class WebWindow(nn.Module):
    def __init__(self, model: Ardy, history_frames: int = 4, num_denoising_steps: int = 10):
        super().__init__()
        assert history_frames % model.num_frames_per_token == 0
        self.model = model
        self.history_frames = history_frames
        self.gen_frames = model.gen_horizon_len
        self.num_denoising_steps = num_denoising_steps
        self.nfpt = model.num_frames_per_token
        self.token_dim = model.denoiser.nframe_root_dim + model.denoiser.latent_embedding_dim
        model.diffusion.ensure_schedule(num_denoising_steps)
        self.register_buffer("map_tensor", model.diffusion.map_tensor.clone(), persistent=False)

    @property
    def noise_shape(self) -> Tuple[int, int, int]:
        return (1, self.gen_frames // self.nfpt, self.token_dim)

    def forward(
        self,
        history: torch.Tensor,  # [1, history_frames, motion_rep_dim], normalized explicit features (world space)
        text_feat: torch.Tensor,  # [1, 1, llm_dim]
        noise: torch.Tensor,  # [1, gen_frames / nfpt, token_dim]
        cfg_weight_text: torch.Tensor,  # [1]
        cfg_weight_cstr: torch.Tensor,  # [1]
    ):
        model, hybrid, motion_rep = self.model, self.model.hybrid, self.model.motion_rep
        device = history.device
        H, G, nfpt = self.history_frames, self.gen_frames, self.nfpt
        T = H + G
        long1 = lambda v: torch.full((1,), v, device=device, dtype=torch.long)  # noqa: E731

        # 1) explicit history -> hybrid tokens, recentered on its last frame (== Ardy._encode_init_history)
        hist_hybrid, _ = hybrid.get_hybrid_motion_from_explicit(
            motion=history, motion_len=long1(H), motion_pad_mask=torch.ones(1, H, device=device, dtype=torch.bool)
        )
        hist_hybrid, center_pos, _ = model._recenter_history(hist_hybrid, long1(H - 1), requantize=False)
        global_transl = center_pos
        first_heading_angle = motion_rep.get_root_heading_angle(motion_rep.unnormalize(history))[:, 0]

        # 2) window masks (== Ardy._generate_window with history_start=0, history_end=H, total=T, no constraints)
        history_len, generation_len, future_len = long1(H), long1(G), long1(0)
        history_mask, generation_mask, future_mask = get_three_mask_from_len(
            history_len, generation_len, future_len, T, device
        )
        history_token_mask, generation_token_mask, future_token_mask = hybrid.convert_frame_mask_to_token_mask(
            history_mask, generation_mask, future_mask, None
        )
        h_tok, g_tok = H // nfpt, G // nfpt
        x = torch.cat([hist_hybrid[:, -h_tok:], noise], dim=1)  # [1, h_tok + g_tok, token_dim]
        text_pad_mask = torch.ones(1, text_feat.shape[1], device=device, dtype=torch.bool)

        # 3) unrolled denoising loop (== Ardy.denoising_step with generation_token_slice)
        for i in reversed(range(self.num_denoising_steps)):
            t = long1(i)
            pred_clean = model.denoiser(
                cfg_weight_text,
                cfg_weight_cstr,
                x,
                history_len,
                generation_len,
                future_len,
                history_mask,
                generation_mask,
                future_mask,
                history_token_mask,
                generation_token_mask,
                future_token_mask,
                text_feat,
                text_pad_mask,
                self.map_tensor[t],
                first_heading_angle,
                None,
                None,
            )
            x_gen = model.sampler(x[:, h_tok : h_tok + g_tok], pred_clean[:, h_tok : h_tok + g_tok], t)
            x = torch.cat([x[:, :h_tok], x_gen], dim=1)

        # 4) back to world space, requantize, decode (== tail of Ardy.autoregressive_step)
        root_motion, latent = hybrid.get_root_and_latent_body_motion_from_hybrid(x)
        root_motion = translate_normalized_root_motion(root_motion, global_transl, motion_rep)
        if model.autoencoder.encode_with_quantization:
            latent = model.autoencoder.requantize(latent)
        x = hybrid.get_hybrid_motion_from_root_and_latent_body_motion(root_motion, latent)
        motion = hybrid.get_explicit_motion_from_hybrid(
            x, torch.ones(1, T, device=device, dtype=torch.bool), long1(T), motion_mask=None
        )  # [1, T, motion_rep_dim], normalized

        # 5) joints for rendering
        out = motion_rep.inverse(motion_rep.unnormalize(motion), is_normalized=False)
        return motion, out["posed_joints"], out["global_rot_mats"]


def reference_window(model: Ardy, history, text_feat, seed: int, cfg_weight, num_denoising_steps: int = 10):
    """The original code path with the same noise WebWindow would get for ``seed`` (see ``draw_noise``)."""
    torch.manual_seed(seed)
    text_pad_mask = torch.ones(1, text_feat.shape[1], device=history.device, dtype=torch.bool)
    return model.autoregressive_step(
        num_frames=model.gen_horizon_len + history.shape[1],  # total frames incl. history
        num_denoising_steps=num_denoising_steps,
        motion_mask=None,
        observed_motion=None,
        cfg_weight=cfg_weight,
        text_feat=text_feat,
        text_pad_mask=text_pad_mask,
        init_history_sequence=history,
    )


def draw_noise(window: WebWindow, seed: int, device) -> torch.Tensor:
    """The noise ``Ardy._generate_window`` draws first after ``torch.manual_seed(seed)``."""
    torch.manual_seed(seed)
    return torch.randn(window.noise_shape, device=device)


def export_web_window(window: WebWindow, path: str, llm_dim: int, opset: int = 17) -> None:
    window.eval()
    device = next(window.parameters()).device
    dummy = (
        torch.zeros(1, window.history_frames, window.model.motion_rep.motion_rep_dim, device=device),
        torch.zeros(1, 1, llm_dim, device=device),
        torch.zeros(window.noise_shape, device=device),
        torch.tensor([3.5], device=device),
        torch.tensor([1.5], device=device),
    )
    # nn.TransformerEncoder's fused fast path (aten::_transformer_encoder_layer_fwd) has no ONNX
    # symbolic; disable it for the trace like scripts/export_onnx.py does.
    fastpath = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        with torch.no_grad():
            _export(window, dummy, path, opset)
    finally:
        torch.backends.mha.set_fastpath_enabled(fastpath)


def _export(window, dummy, path, opset):
    with torch.no_grad():
        torch.onnx.export(
            window,
            dummy,
            path,
            input_names=["history", "text_feat", "noise", "cfg_weight_text", "cfg_weight_cstr"],
            output_names=["motion", "joints", "joint_rotations"],
            opset_version=opset,
            dynamo=False,
            do_constant_folding=True,
        )
