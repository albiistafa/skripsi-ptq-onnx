"""Util bersama: path, preprocessing, pasangan LFW, embedding ONNX Runtime.

Normalisasi identik utk kedua model: (pixel - 127.5)/127.5, RGB.
EdgeFace (Normalize mean=std=0.5) dan GhostFaceNet sama-sama pakai ini.
Layout berbeda: GhostFaceNet NHWC, EdgeFace NCHW (isi piksel identik).
"""
import os
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ONNX_DIR = os.path.join(ROOT, "onnx")
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")

# Aset lokal di new-skripsi/dataset (aligned LFW + daftar pasangan).
ALIGNED = os.path.join(ROOT, "dataset", "lfw_aligned")
PAIRS_TXT = os.path.join(ROOT, "dataset", "pairs.txt")

SIZE = 112


def load_img(path):
    """Baca -> 112x112 RGB float32 ternormalisasi, layout HWC."""
    img = Image.open(path).convert("RGB").resize((SIZE, SIZE))
    x = (np.asarray(img, dtype=np.float32) - 127.5) / 127.5
    return x  # (112,112,3)


def to_layout(x, layout):
    """x: HWC atau NHWC -> NHWC apa adanya, atau NCHW (transpose)."""
    if layout == "nchw":
        ax = (0, 3, 1, 2) if x.ndim == 4 else (2, 0, 1)
        return np.ascontiguousarray(np.transpose(x, ax))
    return x


def img_path(name, idx):
    return os.path.join(ALIGNED, name, f"{name}_{int(idx):04d}.jpg")


def load_pairs():
    """pairs.txt -> (pairs, folds). pairs: (a, b, label). label 1=genuine."""
    with open(PAIRS_TXT) as f:
        lines = f.read().splitlines()
    n_folds, per = (int(x) for x in lines[0].split())
    pairs, folds = [], []
    i = 1
    for fold in range(n_folds):
        for _ in range(per):
            name, a, b = lines[i].split(); i += 1
            pairs.append((img_path(name, a), img_path(name, b), 1))
            folds.append(fold)
        for _ in range(per):
            n1, a, n2, b = lines[i].split(); i += 1
            pairs.append((img_path(n1, a), img_path(n2, b), 0))
            folds.append(fold)
    return pairs, np.array(folds)


# Layout + nama input per model.
MODELS = {
    "ghost": {"layout": "nhwc", "input": "input_face"},
    "edge":  {"layout": "nchw", "input": "input"},
}
