"""Encode a prompt library with the local LLM2Vec encoder for the browser demo (no LLM in the browser).

Usage: python web/build_prompts.py [out_dir] [prompts.txt]
Writes <out_dir>/prompts.json ({"llm_dim", "prompts": [...]}) and prompts.bin (float32 [N, llm_dim]).
"""

import json
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
from ardy.model.load_model import load_text_encoder  # noqa: E402
from ardy.tools import get_default_device  # noqa: E402
from interactive_demo.embedding_cache import CachedTextEncoder, text_encoder_cache_namespace  # noqa: E402

DEFAULT_PROMPTS = [
    "A person is walking.",
    "A person is running.",
    "A person is jogging in a circle.",
    "A person is walking backwards slowly.",
    "A person is standing still.",
    "A person is dancing happily.",
    "A person is jumping up and down.",
    "A person is boxing, throwing punches.",
    "A person is crouching and sneaking forward.",
    "A person is waving with both hands.",
    "A person is kicking with the right leg.",
    "A person is turning around and walking away.",
    "A person is marching.",
    "A person is stretching the arms.",
    "A person is walking like a zombie.",
    "A person is skipping forward.",
    "A person walks in a circle.",
    "A person is doing jumping jacks.",
    "A person is bowing politely.",
    "A person is walking and looking around.",
]

out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
prompts = DEFAULT_PROMPTS if len(sys.argv) < 3 else [l.strip() for l in open(sys.argv[2]) if l.strip()]
os.makedirs(out, exist_ok=True)

encoder = load_text_encoder(mode="local", device=get_default_device())
cached = CachedTextEncoder(encoder, namespace=text_encoder_cache_namespace(encoder))
feats, lengths = cached(prompts)  # [N, L, llm_dim]
assert feats.shape[1] == 1 and all(n == 1 for n in lengths), "expected one pooled token per prompt"
arr = feats[:, 0].to(torch.float32).cpu().numpy()
arr.astype(np.float32).tofile(os.path.join(out, "prompts.bin"))
json.dump({"llm_dim": int(arr.shape[1]), "dtype": "float32", "prompts": prompts}, open(os.path.join(out, "prompts.json"), "w"), indent=0)
print(f"wrote {len(prompts)} prompts x {arr.shape[1]} -> {out}/prompts.bin ({arr.nbytes / 1e6:.1f} MB), prompts.json")
