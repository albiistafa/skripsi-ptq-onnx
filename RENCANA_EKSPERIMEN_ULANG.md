# Rencana Eksperimen Ulang — Komparasi Seluruh Jalur PTQ via ONNX Runtime

Versi ini menstandarkan **seluruh pipeline pada ONNX**: kedua model dikonversi ke ONNX, lalu
seluruh teknik Post-Training Quantization (PTQ) diterapkan menggunakan **ONNX Runtime**
(`onnxruntime.quantization`). TensorFlow/PyTorch hanya berperan sebagai **jembatan konversi**
ke ONNX, bukan tempat kuantisasi.

**Alasan (justifikasi):** kedua model berasal dari framework berbeda (GhostFaceNet=Keras/TF,
EdgeFace=PyTorch). Dengan menyatukan keduanya pada ONNX dan satu runtime (ONNX Runtime),
framework dinetralkan sebagai variabel pengganggu → komparasi adil (apple-to-apple). ONNX
adalah standar terbuka interoperabilitas model, dan ONNX Runtime mendukung kuantisasi
(dynamic, static/QDQ, float16) serta deployment mobile (ONNX Runtime Mobile).

---

## 1. Teknik PTQ yang Diuji (semua via ONNX Runtime)

| Kode | Teknik | Fungsi ONNX Runtime | Yang dikuantisasi | Kalibrasi |
|---|---|---|---|---|
| M0 | FP32 (baseline) | — | — | — |
| M1 | Dynamic INT8 | `quantize_dynamic` | bobot | tidak |
| M2 | Float16 | `convert_float_to_float16` | bobot | tidak |
| M3 | Static INT8 (QDQ) | `quantize_static` | bobot + aktivasi | **ya (200 citra)** |

Setiap teknik diterapkan **identik** pada kedua model.

---

## 2. Prinsip Keadilan (samakan semua faktor)

| Faktor | Disamakan |
|---|---|
| Dataset & alignment | LFW **InsightFace-Aligned**, 6.000 pasangan, 10-fold CV |
| Pra-pemrosesan | crop+resize 112×112, normalisasi identik |
| Subset kalibrasi | **200 citra LFW yang sama** (seed tetap) untuk M3 |
| Toolchain kuantisasi | **ONNX Runtime** untuk kedua model |
| Konfigurasi tiap teknik | identik antar arsitektur |
| Runtime evaluasi & latensi | ONNX Runtime (Android: ONNX Runtime Mobile) |
| Metrik | Akurasi (%), Ukuran (MB), Latensi median 50 iter (ms) |

---

## 3. Lingkungan

```bash
# venv_tf : HANYA untuk konversi GhostFaceNet (.h5 -> ONNX)
python3.10 -m venv venv_tf && source venv_tf/bin/activate
pip install tensorflow==2.15.1 keras_cv_attention_models tf2onnx==1.17.0 "onnx>=1.14"
deactivate

# venv_onnx : konversi EdgeFace + SELURUH PTQ + evaluasi
python3.10 -m venv venv_onnx && source venv_onnx/bin/activate
pip install "numpy<2" onnx==1.15.0 onnxruntime==1.17.3 onnxconverter-common \
            torch torchvision pillow scikit-learn scipy tqdm
```
> Di Colab (Python 3.12): set `os.environ["TF_USE_LEGACY_KERAS"]="1"` sebelum import TF,
> dan `pip uninstall -y jax jaxlib` agar tidak bentrok `ml_dtypes`.

---

## 4. Pipeline

### Tahap 1 — Data (sekali)
- Siapkan LFW **InsightFace-Aligned** + daftar 6.000 pasangan + pembagian 10-fold.
- Pilih **200 citra kalibrasi** (seed tetap) di luar pasangan evaluasi → `calib.npy`,
  normalisasi identik. Siapkan layout: **NHWC** (GhostFaceNet) & **NCHW** (EdgeFace).

### Tahap 2 — Konversi ke ONNX FP32

**GhostFaceNet (venv_tf):**
```python
import tensorflow as tf, tf2onnx
m = tf.keras.models.load_model("models/GN_W1.3_S1_ArcFace_epoch46.h5", compile=False)
spec = (tf.TensorSpec((None,112,112,3), tf.float32, name="input_face"),)
tf2onnx.convert.from_keras(m, input_signature=spec, opset=13,
                           output_path="onnx/ghost_fp32_raw.onnx")
```

**Bersihkan float16 → float32 (khusus GhostFaceNet, venv_onnx):**
```python
import onnx, numpy as np
from onnx import TensorProto, numpy_helper
model = onnx.load("onnx/ghost_fp32_raw.onnx")
# (a) ubah node Cast-to-float16 -> float32
for n in model.graph.node:
    if n.op_type == "Cast":
        for a in n.attribute:
            if a.name == "to" and a.i == TensorProto.FLOAT16:
                a.i = TensorProto.FLOAT
# (b) konversi initializer float16 -> float32
for i, init in enumerate(model.graph.initializer):
    if init.data_type == TensorProto.FLOAT16:
        arr = numpy_helper.to_array(init).astype(np.float32)
        model.graph.initializer[i].CopyFrom(numpy_helper.from_array(arr, init.name))
onnx.save(model, "onnx/ghost_fp32.onnx")   # FP32 bersih, siap dikuantisasi
```
> Ini setara dengan yang sudah berhasil di skrip `03c_step2_quantize_onnx.py`-mu.

**EdgeFace (venv_onnx):**
```python
import torch
model = torch.hub.load("idiap/EdgeFace", "edgeface_xs_gamma_06", source="github")  # sesuaikan nama
model.eval()
dummy = torch.randn(1, 3, 112, 112)   # NCHW
torch.onnx.export(model, dummy, "onnx/edge_fp32.onnx",
                  input_names=["input"], output_names=["emb"], opset_version=13)
```

### Tahap 3 — Terapkan SETIAP teknik PTQ (ONNX Runtime, identik kedua model)

**M1 — Dynamic INT8:**
```python
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic("onnx/ghost_fp32.onnx", "onnx/ghost_m1_dyn.onnx", weight_type=QuantType.QInt8)
quantize_dynamic("onnx/edge_fp32.onnx",  "onnx/edge_m1_dyn.onnx",  weight_type=QuantType.QInt8)
```

**M2 — Float16:**
```python
import onnx
from onnxconverter_common import float16
for src, dst in [("onnx/ghost_fp32.onnx","onnx/ghost_m2_fp16.onnx"),
                 ("onnx/edge_fp32.onnx","onnx/edge_m2_fp16.onnx")]:
    onnx.save(float16.convert_float_to_float16(onnx.load(src)), dst)
```

**M3 — Static INT8 (QDQ, butuh kalibrasi):**
```python
from onnxruntime.quantization import quantize_static, QuantType, QuantFormat, CalibrationDataReader
import numpy as np

class Reader(CalibrationDataReader):
    def __init__(self, npy, input_name):
        self.data = np.load(npy).astype("float32"); self.input_name = input_name; self.i = 0
    def get_next(self):
        if self.i >= len(self.data): return None
        x = self.data[self.i:self.i+1]; self.i += 1
        return {self.input_name: x}

def static_q(fp32, out, calib, inp):
    quantize_static(fp32, out, calibration_data_reader=Reader(calib, inp),
                    quant_format=QuantFormat.QDQ, per_channel=True,
                    weight_type=QuantType.QInt8, activation_type=QuantType.QInt8)

static_q("onnx/ghost_fp32.onnx", "onnx/ghost_m3_qdq.onnx", "data/calib_nhwc.npy", "input_face")
static_q("onnx/edge_fp32.onnx",  "onnx/edge_m3_qdq.onnx",  "data/calib_nchw.npy", "input")
```

**Catat status tiap konversi** (berhasil/gagal + pesan) ke `results/compatibility.csv`.

### Tahap 4 — Evaluasi Akurasi (ONNX Runtime, protokol identik)
```python
import onnxruntime as ort, numpy as np
def embed(model_path, img):           # img sesuai layout model, sudah dinormalisasi
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    return sess.run(None, {name: img.astype("float32")})[0]
```
Untuk tiap model: ekstrak embedding semua citra → L2-normalize → cosine similarity tiap
pasangan → 10-fold (threshold EER per fold) → mean ± std. Hitung Δacc vs FP32.
Simpan `results/accuracy_<model>_<teknik>.csv`.

> **Wajib pakai dataset aligned.** Versi deepfunneled/resize-utuh menurunkan akurasi drastis (~73%).

### Tahap 5 — Ukuran Model
Catat ukuran berkas `.onnx` tiap varian (MB) + rasio kompresi terhadap FP32.

### Tahap 6 — Latensi On-Device (Android, ONNX Runtime Mobile)
- Bungkus tiap model `.onnx` dalam aplikasi testbed berbasis **ONNX Runtime Mobile**.
- Tiap model: 50 iterasi pada citra sama → **median**. Satu perangkat untuk semua.
- Catat spesifikasi perangkat. Simpan `results/latency.csv`.

### Tahap 7 — Matriks & Analisis
Susun Tabel A (kompatibilitas) + Tabel B (akurasi/ukuran/latensi per teknik) + grafik trade-off.

---

## 5. Catatan Teknis Penting
1. **GhostFaceNet wajib dibersihkan float16 → float32** sebelum kuantisasi (Tahap 2). EdgeFace
   tidak perlu.
2. **Layout berbeda:** GhostFaceNet NHWC (112,112,3), EdgeFace NCHW (3,112,112). Sesuaikan
   `calib.npy` & preprocessing, tapi **isi piksel & normalisasi tetap identik**.
3. **Semua kuantisasi & evaluasi via ONNX Runtime** — jangan campur TFLite.
4. **Konfigurasi tiap teknik identik** antar kedua model (mis. M3 sama-sama QDQ per-channel).

---

## 6. Matriks Hasil yang Ditargetkan

**Tabel A — Kompatibilitas (berhasil / gagal):**

| Teknik | EdgeFace | GhostFaceNet |
|---|---|---|
| M1 Dynamic INT8 | … | … |
| M2 Float16 | … | … |
| M3 Static INT8 (QDQ) | … | … |

**Tabel B — Akurasi / Ukuran / Latensi:**

| Model | Teknik | Akurasi (mean±std) | Δacc | Ukuran (MB) | Latensi (ms) |
|---|---|---|---|---|---|
| EdgeFace | FP32 | … | — | … | … |
| EdgeFace | Dynamic INT8 | … | … | … | … |
| EdgeFace | Float16 | … | … | … | … |
| EdgeFace | Static INT8 | … | … | … | … |
| GhostFaceNet | FP32 | … | — | … | … |
| GhostFaceNet | Dynamic INT8 | … | … | … | … |
| GhostFaceNet | Float16 | … | … | … | … |
| GhostFaceNet | Static INT8 | … | … | … | … |

---

## 7. Checklist
- [ ] Kedua model dikonversi ke ONNX FP32 (GhostFaceNet sudah dibersihkan float16)
- [ ] Setiap teknik (M1–M3) dikenakan ke kedua model dengan konfigurasi identik
- [ ] Status berhasil/gagal tercatat (Tabel A)
- [ ] Evaluasi via ONNX Runtime, dataset aligned, 10-fold
- [ ] Kalibrasi: 200 citra sama (seed tetap) untuk M3
- [ ] Latensi: satu perangkat, 50 iterasi, median, ONNX Runtime Mobile

---

## 8. Catatan untuk Sidang
- **Justifikasi ONNX:** standar terbuka interoperabilitas model; menyatukan dua arsitektur
  lintas-framework pada satu runtime menghilangkan framework sebagai variabel pengganggu →
  komparasi adil. ONNX Runtime mendukung PTQ (dynamic/static/float16) dan deployment mobile.
- **Konfirmasi ke pembimbing:** target deployment menjadi **ONNX Runtime Mobile** (bukan TFLite).
- **Temuan kompatibilitas** tiap teknik × arsitektur = kontribusi penelitian.

---

## Referensi pendukung
- ONNX — Open standard for ML interoperability: https://github.com/onnx/onnx
- ONNX Runtime (inference + quantization + mobile): https://onnxruntime.ai/
- Krishnamoorthi (2018), Quantization whitepaper: arXiv:1806.08342
- Jacob dkk. (2018), Integer-Arithmetic-Only Inference, CVPR
- Nagel dkk. (2021), A White Paper on Neural Network Quantization: arXiv:2106.08295
