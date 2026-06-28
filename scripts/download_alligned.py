"""
Script 01c: Download InsightFace-Aligned LFW (lfw.bin)
=======================================================
InsightFace menyediakan LFW yang sudah di-align menggunakan RetinaFace
+ 5-point landmark warp ke 112x112 - format yang sama dengan training
GhostFaceNet dan EdgeFace.

Format lfw.bin:
  - pickle file berisi (bins, issame_list)
  - bins: list of 12000 image bytes (2 gambar x 6000 pairs)
  - issame_list: list of 6000 bool (True = same person)

Sumber: https://github.com/deepinsight/insightface/tree/master/recognition
Alternatif: https://github.com/ZhaoJ9014/face.evoLVe.PyTorch

Jalankan: python scripts/01c_download_aligned_lfw.py
"""

import os
import sys
import pickle
import numpy as np
from PIL import Image
import io

# -- Konfigurasi ----------------------------------------------
LFW_BIN_PATH = "data/aligned/lfw.bin"
PAIRS_OUTPUT = "data/pairs/eval_pairs_aligned.pkl"
IMG_SAVE_DIR = "data/lfw-aligned"   # simpan gambar hasil extract
# -------------------------------------------------------------


def download_lfw_bin():
    """Download lfw.bin dari sumber yang tersedia."""
    os.makedirs(os.path.dirname(LFW_BIN_PATH), exist_ok=True)

    if os.path.exists(LFW_BIN_PATH):
        size_mb = os.path.getsize(LFW_BIN_PATH) / (1024 * 1024)
        print(f"[INFO] lfw.bin sudah ada: {LFW_BIN_PATH} ({size_mb:.1f} MB)")
        return True

    print("[INFO] Mencoba download lfw.bin...")

    # Opsi 1: via insightface package
    try:
        import insightface
        from insightface.utils import face_align
        print("       insightface tersedia")
    except ImportError:
        print("       insightface tidak tersedia (install: pip install insightface)")

    # Opsi 2: via gdown (Google Drive InsightFace mirror)
    gdrive_id = "1WwFRRiMOAz6gMt7F3v11ZW_gzI0XuMi0"

    try:
        import gdown
        print(f"       Downloading via gdown (Google Drive ID: {gdrive_id})...")
        gdown.download(
            f"https://drive.google.com/uc?id={gdrive_id}",
            LFW_BIN_PATH,
            quiet=False
        )
        if os.path.exists(LFW_BIN_PATH):
            size_mb = os.path.getsize(LFW_BIN_PATH) / (1024 * 1024)
            print(f"       Downloaded: {size_mb:.1f} MB")
            return True
    except ImportError:
        print("       gdown tidak tersedia, install: pip install gdown")
    except Exception as e:
        print(f"       gdown gagal: {e}")

    # Opsi 3: via urllib langsung
    alt_urls = [
        "https://drive.google.com/uc?export=download&id=1WwFRRiMOAz6gMt7F3v11ZW_gzI0XuMi0",
    ]

    import urllib.request
    for url in alt_urls:
        try:
            print(f"       Mencoba: {url[:60]}...")
            urllib.request.urlretrieve(url, LFW_BIN_PATH)
            if os.path.getsize(LFW_BIN_PATH) > 1000:
                size_mb = os.path.getsize(LFW_BIN_PATH) / (1024 * 1024)
                print(f"       Downloaded: {size_mb:.1f} MB")
                return True
        except Exception as e:
            print(f"       Gagal: {e}")

    print("""
[ERROR] Tidak dapat download lfw.bin otomatis.
Download manual dari salah satu sumber berikut:

1. InsightFace (Google Drive):
   https://drive.google.com/file/d/1WwFRRiMOAz6gMt7F3v11ZW_gzI0XuMi0/view

2. face.evoLVe.PyTorch:
   https://github.com/ZhaoJ9014/face.evoLVe.PyTorch#data-zoo

Setelah download, letakkan di: data/aligned/lfw.bin
Kemudian jalankan script ini kembali.
""")
    return False


def extract_lfw_bin(save_images: bool = True):
    """
    Baca lfw.bin dan ekstrak:
    - Gambar ke data/lfw-aligned/
    - Pairs ke data/pairs/eval_pairs_aligned.pkl
    """
    print(f"\n[INFO] Membaca {LFW_BIN_PATH}...")
    with open(LFW_BIN_PATH, "rb") as f:
        bins, issame_list = pickle.load(f, encoding="bytes")

    print(f"       Total images  : {len(bins)}")
    print(f"       Total pairs   : {len(issame_list)}")
    print(f"       Same pairs    : {sum(issame_list)}")
    print(f"       Diff pairs    : {sum(not x for x in issame_list)}")

    # Verifikasi format
    test_img = Image.open(io.BytesIO(bins[0])).convert("RGB")
    print(f"       Image size    : {test_img.size}")  # harusnya (112, 112)

    if save_images:
        os.makedirs(IMG_SAVE_DIR, exist_ok=True)
        print(f"\n[INFO] Menyimpan gambar ke {IMG_SAVE_DIR}/...")

    # Build pairs dengan path gambar
    pairs_list = []

    for idx, (img_bytes, is_same) in enumerate(zip(
        zip(bins[0::2], bins[1::2]),  # pasangkan 2 gambar per pair
        issame_list
    )):
        img1_bytes, img2_bytes = img_bytes
        p1 = os.path.join(IMG_SAVE_DIR, f"pair{idx:04d}_a.jpg")
        p2 = os.path.join(IMG_SAVE_DIR, f"pair{idx:04d}_b.jpg")

        if save_images:
            Image.open(io.BytesIO(img1_bytes)).convert("RGB").save(p1)
            Image.open(io.BytesIO(img2_bytes)).convert("RGB").save(p2)

        pairs_list.append((p1, p2, int(is_same)))

        if idx % 1000 == 0:
            print(f"       Extracted {idx}/{len(issame_list)}...")

    # Simpan pkl
    os.makedirs(os.path.dirname(PAIRS_OUTPUT), exist_ok=True)
    with open(PAIRS_OUTPUT, "wb") as f:
        pickle.dump({
            "pairs"    : pairs_list,
            "n_folds"  : 10,
            "source"   : "insightface_aligned_lfw",
            "img_size" : 112,
        }, f)

    print(f"\nEkstraksi selesai!")
    print(f"   Gambar    : {IMG_SAVE_DIR}/ ({len(pairs_list)*2} files)")
    print(f"   Pairs pkl : {PAIRS_OUTPUT}")
    print(f"\n[NEXT] Jalankan evaluasi dengan aligned pairs:")
    print(f"   python scripts/04b_evaluate_aligned.py")


def main():
    print("=" * 60)
    print("Download & Extract InsightFace-Aligned LFW")
    print("=" * 60)

    if not download_lfw_bin():
        sys.exit(1)

    extract_lfw_bin(save_images=True)


if __name__ == "__main__":
    main()
