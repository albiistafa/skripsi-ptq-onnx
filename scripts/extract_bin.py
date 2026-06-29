"""Ekstrak .bin verifikasi InsightFace apa pun (lfw/cfp_fp/agedb) -> gambar + pkl.

Format .bin (mxnet): pickle (bins, issame).
  bins   = 2*N byte-gambar (2 per pasang), issame = N bool (True=orang sama).
  pasang i: img1=bins[2i], img2=bins[2i+1], label=int(issame[i]); fold = i // (N//10).

Pakai: python scripts/extract_bin.py <path.bin> <dataset_key>
  dataset_key: lfw | cfpfp   (menentukan nama output)
"""
import io
import os
import pickle
import sys
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# dataset_key -> (folder gambar, pkl pasangan, jumlah pasang yg diharapkan)
# Keenam set verifikasi InsightFace (faces_webface_112x112).
KEYS = {
    "lfw":     ("dataset/lfwbin",    "data/pairs_lfwbin.pkl",  6000),
    "cfpfp":   ("dataset/cfpfp_bin", "data/pairs_cfpfp.pkl",   7000),
    "cfpff":   ("dataset/cfpff_bin", "data/pairs_cfpff.pkl",   7000),
    "agedb30": ("dataset/agedb_bin", "data/pairs_agedb30.pkl", 6000),
    "calfw":   ("dataset/calfw_bin", "data/pairs_calfw.pkl",   6000),
    "cplfw":   ("dataset/cplfw_bin", "data/pairs_cplfw.pkl",   6000),
}


def main(bin_path, key):
    if key not in KEYS:
        raise SystemExit(f"dataset_key tak dikenal: {key} (pilih {list(KEYS)})")
    img_dir, pkl_path, expect = KEYS[key]
    img_dir = os.path.join(ROOT, img_dir)
    pkl_path = os.path.join(ROOT, pkl_path)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(os.path.dirname(pkl_path), exist_ok=True)

    if os.path.getsize(bin_path) < 1_000_000:
        raise SystemExit(f"{bin_path} terlalu kecil ({os.path.getsize(bin_path)} B)")

    with open(bin_path, "rb") as f:
        bins, issame = pickle.load(f, encoding="bytes")

    n_pairs = len(issame)
    per_fold = n_pairs // 10
    print(f"{key}: bins={len(bins)} pasang={n_pairs} (harapan {expect}) per_fold={per_fold}")
    assert len(bins) == 2 * n_pairs, "jumlah bins != 2*issame"

    pairs, folds, sizes = [], [], set()
    for i in range(n_pairs):
        pa = os.path.join(img_dir, f"pair{i:05d}_a.jpg")
        pb = os.path.join(img_dir, f"pair{i:05d}_b.jpg")
        for path, raw in ((pa, bins[2 * i]), (pb, bins[2 * i + 1])):
            img = Image.open(io.BytesIO(bytes(raw))).convert("RGB")
            sizes.add(img.size)
            img.save(path, quality=100)
        pairs.append((pa, pb, int(issame[i])))
        folds.append(i // per_fold)

    with open(pkl_path, "wb") as f:
        pickle.dump({"pairs": pairs, "folds": folds}, f)

    genuine = sum(l for *_, l in pairs)
    print(f"pasang {len(pairs)} | genuine {genuine} | impostor {len(pairs)-genuine}")
    print(f"ukuran gambar {sizes} (harus 112x112) | folds {sorted(set(folds))}")
    print("->", pkl_path)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("pakai: python scripts/extract_bin.py <path.bin> <lfw|cfpfp>")
    main(sys.argv[1], sys.argv[2])
