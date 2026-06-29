# Ringkasan Akurasi: 6 Benchmark x 8 Model (ONNX Runtime)

Satu set model (dikuantisasi+dikalibrasi sekali dengan LFW) diuji pada 6 benchmark verifikasi InsightFace. Akurasi 10-fold (%).

## EdgeFace

| Teknik | LFW | CFP-FP | CFP-FF | AgeDB-30 | CALFW | CPLFW |
|---|---|---|---|---|---|---|
| FP32 | 99.77 | 95.89 | 99.54 | 96.85 | 95.62 | 92.27 |
| M1 Dyn-INT8 | 99.60 | 94.40 | 99.46 | 95.73 | 95.08 | 90.95 |
| M2 FP16 | 99.77 | 95.87 | 99.54 | 96.85 | 95.62 | 92.27 |
| M3 Static-INT8 | 99.58 | 94.29 | 99.39 | 95.45 | 95.32 | 91.07 |

## GhostFaceNet

| Teknik | LFW | CFP-FP | CFP-FF | AgeDB-30 | CALFW | CPLFW |
|---|---|---|---|---|---|---|
| FP32 | 99.78 | 93.67 | 99.70 | 97.52 | 95.80 | 90.03 |
| M1 Dyn-INT8 | 99.62 | 92.00 | 99.69 | 96.60 | 95.58 | 89.40 |
| M2 FP16 | 99.78 | 93.71 | 99.70 | 97.45 | 95.80 | 90.10 |
| M3 Static-INT8 | 98.83 | 86.96 | 98.93 | 92.35 | 93.18 | 82.17 |
