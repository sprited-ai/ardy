"""Pack the Core skin (LBS data used by the viser demo) for the browser: skin_<skeleton>.json + .bin.

Usage: python web/build_skin.py [out_dir] [skeleton_folder]
The .bin holds float32/uint16/uint8 arrays back to back; the .json lists dtype, shape and byte offset of each.
"""

import json
import os
import sys

import numpy as np

out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
folder = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ardy", "assets", "skeletons", "cskel27")
name = os.path.basename(folder.rstrip("/"))
d = np.load(os.path.join(folder, "skin_standard.npz"), allow_pickle=True)
arrays = {
    "bind_vertices": d["bind_vertices"].astype(np.float32),                      # [V, 3]
    "faces": d["faces"].astype(np.uint16 if d["faces"].max() < 65535 else np.uint32),  # [F, 3]
    "bind_rig_transform_inv": np.linalg.inv(d["bind_rig_transform"].astype(np.float64)).astype(np.float32),  # [J, 4, 4]
    "lbs_indices": d["lbs_indices"].astype(np.uint8),                             # [V, W]
    "lbs_weights": d["lbs_weights"].astype(np.float32),                           # [V, W]
}
blob, manifest, offset = bytearray(), {"skeleton": name, "rig_joint_names": [str(s) for s in d["rig_joint_names"]], "arrays": {}}, 0
for key, arr in arrays.items():
    arr = np.ascontiguousarray(arr)
    if offset % 4:  # keep float32 views aligned
        pad = 4 - offset % 4; blob += b"\0" * pad; offset += pad
    manifest["arrays"][key] = {"dtype": str(arr.dtype), "shape": list(arr.shape), "offset": offset, "length": int(arr.size)}
    blob += arr.tobytes(); offset += arr.nbytes
os.makedirs(out, exist_ok=True)
open(os.path.join(out, f"skin_{name}.bin"), "wb").write(blob)
json.dump(manifest, open(os.path.join(out, f"skin_{name}.json"), "w"))
print(f"wrote skin_{name}.bin ({len(blob) / 1e6:.2f} MB): " + ", ".join(f"{k} {v['shape']}" for k, v in manifest["arrays"].items()))
