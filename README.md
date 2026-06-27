# Komparasi PTQ EdgeFace vs GhostFaceNet via ONNX Runtime

Eksperimen skripsi: bandingkan tiga teknik Post-Training Quantization (PTQ) pada dua
arsitektur pengenalan wajah, **disatukan di ONNX Runtime** agar framework netral
(apple-to-apple). Lihat [RENCANA_EKSPERIMEN_ULANG.md](RENCANA_EKSPERIMEN_ULANG.md).

## Model & teknik
- **GhostFaceNet** (Keras/TF, NHWC) — base di `base_model/GN_*.h5`
- **EdgeFace** (`otroshi/edgeface`, edgeface_s_gamma_05, PyTorch, NCHW) — base di `base_model/*.pt`
- **M1** Dynamic INT8 · **M2** Float16 · **M3** Static INT8 (QDQ, kalibrasi 200 citra)

> M1 memakai bobot **QUInt8** (bukan QInt8): `ConvInteger` di ORT CPU butuh bobot
> unsigned, jika tidak inferensi gagal pada model padat-Conv.

## Struktur
```
base_model/   model asli (GhostFaceNet .h5, EdgeFace .pt)
onnx/         hasil konversi + kuantisasi (*.onnx)
data/         calib_nhwc.npy, calib_nchw.npy (200 citra, seed 42)
dataset/      pairs.txt (gambar aligned LFW tidak di-track, lihat .gitignore)
scripts/      pipeline
results/      compatibility.csv (Tabel A), accuracy.csv (Tabel B)
```

## Menjalankan
```bash
python3 -m venv venv_onnx && ./venv_onnx/bin/pip install "numpy<2" torch torchvision onnx onnxruntime onnxconverter-common pillow scikit-learn tqdm timm
python3 -m venv venv_tf  && ./venv_tf/bin/pip install "numpy<2" tensorflow==2.16.2 tf_keras==2.16.0 tf2onnx "onnx>=1.14"

./venv_onnx/bin/python scripts/convert_edge.py            # EdgeFace -> ONNX
./venv_tf/bin/python   scripts/convert_ghost.py           # GhostFaceNet -> ONNX
./venv_onnx/bin/python scripts/build_calib.py             # set kalibrasi
./venv_onnx/bin/python scripts/quantize.py                # M1/M2/M3 -> Tabel A
PYTHONPATH=scripts ./venv_onnx/bin/python scripts/evaluate.py  # 10-fold -> Tabel B
```

Dataset: LFW aligned (InsightFace/RetinaFace, 112×112), protokol 6.000 pasangan 10-fold.
