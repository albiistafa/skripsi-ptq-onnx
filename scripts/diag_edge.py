"""Diagnosis EdgeFace: NaN/Inf/degenerate vs degradasi asli.

Cek embedding 5 gambar utk edge fp32 / M2 fp16 / M3 qdq (NCHW, input "input").
"""
import glob
import os
import numpy as np
import onnxruntime as ort

from common import load_img, to_layout

IMG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "data", "lfw-aligned")
paths = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")))[:5]
print(f"{len(paths)} gambar uji dari {IMG_DIR}\n")

X = to_layout(np.stack([load_img(p) for p in paths]), "nchw").astype(np.float32)

MODELS = ["edge_fp32.onnx", "edge_M2_fp16.onnx", "edge_M3_qdq.onnx"]
for name in MODELS:
    path = os.path.join("onnx", name)
    if not os.path.exists(path):
        print(f"{name}: TIDAK ADA"); continue
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    inp = s.get_inputs()[0]
    dt = np.float16 if "float16" in inp.type else np.float32
    emb = s.run(None, {inp.name: X.astype(dt)})[0].astype(np.float32)
    norms = np.linalg.norm(emb, axis=1)
    inter_std = float(emb.std(axis=0).mean())  # variasi antar-sampel
    print(f"== {name} (in {inp.type}) ==")
    print(f"   NaN={np.isnan(emb).any()}  Inf={np.isinf(emb).any()}")
    print(f"   L2 norms     = {np.round(norms, 3)}")
    print(f"   std antar-sampel (mean per-dim) = {inter_std:.6f}"
          f"  {'<-- KONSTAN/degenerate' if inter_std < 1e-4 else ''}")
    # cosine antar 2 sampel berbeda (harusnya < 1 utk orang beda)
    a = emb[0] / (norms[0] + 1e-10); b = emb[1] / (norms[1] + 1e-10)
    print(f"   cos(sample0,sample1) = {float(np.dot(a, b)):.4f}\n")
