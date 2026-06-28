# Perbandingan PTQ: Sebelum vs Sesudah Mitigasi (LFW lfw.bin, 6000 pasang, 10-fold)

Hanya **EdgeFace M2 & M3** yang bermasalah; sisanya tak berubah. "Sebelum" = hasil
pada dataset lfw.bin sebelum mitigasi EdgeFace.

| Model | Teknik | Akurasi SEBELUM | Akurasi SESUDAH | Δ vs FP32 | Ukuran (MB) | Status |
|---|---|---|---|---|---|---|
| GhostFaceNet | FP32 | 99.78% | 99.78% | — | 16.51 | tak berubah |
| GhostFaceNet | M1 Dynamic INT8 | 99.62% | 99.62% | -0.17 | 4.62 | tak berubah |
| GhostFaceNet | M2 Float16 | 99.78% | 99.78% | +0.00 | 8.34 | tak berubah |
| GhostFaceNet | M3 Static INT8 | 98.83% | 98.83% | -0.95 | 4.86 | tak berubah |
| EdgeFace | FP32 | 99.77% | 99.77% | — | 14.85 | tak berubah |
| EdgeFace | M1 Dynamic INT8 | 99.60% | 99.60% | -0.17 | 4.26 | tak berubah |
| **EdgeFace** | **M2 Float16** | **50.00% (NaN)** | **99.77%** | **+0.00** | 8.42 | **DIPERBAIKI** |
| **EdgeFace** | **M3 Static INT8** | **52.35% (kolaps)** | **99.58%** | **-0.18** | 13.85 | **DIPERBAIKI** |

## Penjelasan masalah & perbaikan (hanya EdgeFace)

### M2 Float16: 50.00% → 99.77%
- **Sebelum:** embedding seluruhnya **NaN**. Attention XCA (cross-covariance) EdgeFace
  membagi dengan norma-L2; di fp16 pembilang/penyebut overflow → `xca/Div` = Inf → NaN.
  Library standar (`onnxconverter_common`, ORT transformers) meng-cast SEMUA edge ke fp16
  sehingga tak bisa melokalisir; akurasi jatuh ke 50% (tebak acak).
- **Perbaikan:** konverter mixed-precision manual (`scripts/mixed_fp16.py`) yang
  **mempertahankan node attention XCA + positional-embedding tetap fp32**, sisanya fp16.
  Hasil identik FP32 (cosine vs fp32 = 1.0000).

### M3 Static INT8 (QDQ): 52.35% → 99.58%
- **Sebelum:** embedding **kolaps** ke ~satu vektor konstan (cos antar orang berbeda 0.985
  vs 0.70 di FP32). Penyebab: kuantisasi **MatMul transformer** meruntuhkan ruang embedding.
  Bukan bug kalibrasi (calib `(200,3,112,112)` float32 `[-1,1]` sudah benar) dan tak teratasi
  oleh per_channel/per_tensor, MinMax/Percentile/Entropy, maupun exclude-attention.
- **Perbaikan:** **kuantisasi Conv saja**; MatMul/Gemm/LayerNorm/Softmax transformer
  dikecualikan (tetap fp32). Akurasi pulih ke 99.58%.
- **Konsekuensi:** karena bobot EdgeFace mayoritas di MatMul transformer (yang kini fp32),
  kompresi M3 kecil (14.85→13.85 MB, 1.07×). Ini **trade-off arsitektural**, bukan kegagalan.

## Temuan untuk sidang
EdgeFace (hybrid CNN-Transformer/XCiT) **rapuh terhadap kuantisasi agresif**:
fp16 butuh attention dipertahankan fp32; static INT8 hanya aman pada bagian Conv.
GhostFaceNet (CNN murni) tahan kuantisasi penuh (static INT8 3.4× kompresi, -0.95 poin).
Perbedaan ini = kontribusi penelitian: pemilihan teknik PTQ harus sadar-arsitektur.
