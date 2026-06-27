"""200 citra kalibrasi (seed tetap), DI LUAR pasangan evaluasi.

Simpan calib_nhwc.npy (GhostFaceNet) & calib_nchw.npy (EdgeFace) -- isi piksel
identik, hanya layout beda.
"""
import glob
import os
import numpy as np

from common import ALIGNED, DATA_DIR, load_pairs, load_img, to_layout

N = 200
SEED = 42

os.makedirs(DATA_DIR, exist_ok=True)

pairs, _ = load_pairs()
eval_imgs = {p for a, b, _ in pairs for p in (a, b)}

all_imgs = sorted(glob.glob(os.path.join(ALIGNED, "*", "*.jpg")))
pool = [p for p in all_imgs if p not in eval_imgs]
print(f"total {len(all_imgs)} | di luar eval {len(pool)}")

rng = np.random.default_rng(SEED)
pick = rng.choice(len(pool), size=N, replace=False)
paths = [pool[i] for i in sorted(pick)]

x = np.stack([load_img(p) for p in paths])  # (200,112,112,3)
np.save(os.path.join(DATA_DIR, "calib_nhwc.npy"), to_layout(x, "nhwc"))
np.save(os.path.join(DATA_DIR, "calib_nchw.npy"), to_layout(x, "nchw"))
print("tersimpan calib_nhwc.npy", to_layout(x, "nhwc").shape,
      "| calib_nchw.npy", to_layout(x, "nchw").shape)
