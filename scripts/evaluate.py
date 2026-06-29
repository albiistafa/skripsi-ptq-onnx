"""Evaluasi 10-fold CV (LFW / CFP-FP) + ukuran + latensi CPU utk 8 model.

SATU model, banyak uji: model onnx/*.onnx (sudah dikuantisasi+dikalibrasi sekali
dgn LFW) dijalankan apa adanya pada pasangan benchmark berbeda. TIDAK ada
kuantisasi/kalibrasi ulang per-benchmark.

Protokol: tiap fold threshold dari 9 fold lain, diuji pada fold tertahan.
Cosine distance = 1 - cos_sim (embedding L2-norm).
Output: results/accuracy_<dataset>.csv.

Latensi = CPU desktop (median 50 iter) proxy; latensi Android (ORT Mobile) terpisah.

Pakai: PYTHONPATH=scripts python scripts/evaluate.py --dataset {lfw|cfpfp}
"""
import argparse
import csv
import os
import pickle
import sys
import time
import numpy as np
import onnxruntime as ort

from common import (ROOT, ONNX_DIR, RESULTS_DIR, MODELS, load_img, to_layout)

# dataset -> pkl pasangan (dihasilkan scripts/extract_bin.py).
PAIRS_PKL = {
    "lfw":     os.path.join(ROOT, "data", "pairs_lfwbin.pkl"),
    "cfpfp":   os.path.join(ROOT, "data", "pairs_cfpfp.pkl"),
    "cfpff":   os.path.join(ROOT, "data", "pairs_cfpff.pkl"),
    "agedb30": os.path.join(ROOT, "data", "pairs_agedb30.pkl"),
    "calfw":   os.path.join(ROOT, "data", "pairs_calfw.pkl"),
    "cplfw":   os.path.join(ROOT, "data", "pairs_cplfw.pkl"),
}


def load_pairs(dataset):
    """Baca pairs_<dataset>.pkl -> (pairs, folds). pairs: (a,b,label) 1=genuine."""
    with open(PAIRS_PKL[dataset], "rb") as f:
        d = pickle.load(f)
    return d["pairs"], np.array(d["folds"])


class Tee:
    """Cetak ke konsol DAN results/eval_log.txt sekaligus (line-flushed)."""
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.term = sys.__stdout__
        self.log = open(path, "w", buffering=1)

    def write(self, s):
        self.term.write(s); self.term.flush()
        self.log.write(s); self.log.flush()

    def flush(self):
        self.term.flush(); self.log.flush()


VARIANTS = ["fp32", "M1_dyn", "M2_fp16", "M3_qdq"]
NP_DTYPE = {"tensor(float)": np.float32, "tensor(float16)": np.float16}


def session(path):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1  # latensi single-thread konsisten
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def embed_all(sess, paths, layout):
    inp = sess.get_inputs()[0]
    dt = NP_DTYPE.get(inp.type, np.float32)
    out = {}
    for k, p in enumerate(paths):
        x = to_layout(load_img(p)[None], layout).astype(dt)
        v = sess.run(None, {inp.name: x})[0][0].astype(np.float32)
        out[p] = v / (np.linalg.norm(v) + 1e-10)
        if (k + 1) % 2000 == 0:
            print(f"     embedding {k+1}/{len(paths)}")
    return out


def best_threshold(dist, label):
    ths = np.unique(dist)
    accs = [((dist < t) == label).mean() for t in ths]
    return ths[int(np.argmax(accs))]


def cross_validate(dist, label, folds):
    accs = []
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        t = best_threshold(dist[tr], label[tr])
        accs.append(((dist[te] < t) == label[te]).mean())
    return np.mean(accs), np.std(accs)


def latency_ms(sess, layout, iters=50):
    inp = sess.get_inputs()[0]
    dt = NP_DTYPE.get(inp.type, np.float32)
    shape = [1, 112, 112, 3] if layout == "nhwc" else [1, 3, 112, 112]
    x = np.random.uniform(-1, 1, shape).astype(dt)
    sess.run(None, {inp.name: x})  # warmup
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        sess.run(None, {inp.name: x})
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts))


def main(dataset):
    sys.stdout = Tee(os.path.join(RESULTS_DIR, f"eval_log_{dataset}.txt"))
    pairs, folds = load_pairs(dataset)
    pairs = [(a, b, l) for (a, b, l) in pairs if os.path.exists(a) and os.path.exists(b)]
    uniq = sorted({p for a, b, _ in pairs for p in (a, b)})
    print(f"[{dataset}] pasangan {len(pairs)} | citra unik {len(uniq)}\n")

    rows = []
    base_acc = {}
    for mname, cfg in MODELS.items():
        layout = cfg["layout"]
        for var in VARIANTS:
            path = os.path.join(ONNX_DIR, f"{mname}_{var}.onnx")
            if not os.path.exists(path):
                print(f"  (lewati) {mname} {var}: tidak ada")
                continue
            mb = os.path.getsize(path) / 1e6
            try:
                sess = session(path)
                emb = embed_all(sess, uniq, layout)
                dist, label, fold = zip(*[
                    (1.0 - float(np.dot(emb[a], emb[b])), l, f)
                    for (a, b, l), f in zip(pairs, folds)])
                (mean, std) = cross_validate(np.array(dist), np.array(label),
                                             np.array(fold))
                lat = latency_ms(sess, layout)
            except Exception as e:  # noqa: BLE001 -- catat sbg temuan kompatibilitas
                print(f"  GAGAL {mname:5s} {var:8s}: {str(e)[:120]}")
                rows.append((mname, var, "GAGAL", "", "", f"{mb:.2f}", ""))
                continue
            if var == "fp32":
                base_acc[mname] = mean
            dacc = (mean - base_acc[mname]) * 100
            rows.append((mname, var, f"{mean*100:.2f}", f"{std*100:.2f}",
                         f"{dacc:+.2f}", f"{mb:.2f}", f"{lat:.2f}"))
            print(f"  {mname:5s} {var:8s} acc {mean*100:5.2f}% +/- {std*100:4.2f}"
                  f"  dacc {dacc:+5.2f}  {mb:5.2f}MB  {lat:6.2f}ms")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, f"accuracy_{dataset}.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "teknik", "akurasi_%", "std_%", "dacc_poin",
                    "ukuran_MB", "latensi_cpu_ms"])
        w.writerows(rows)
    print(f"\n-> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(PAIRS_PKL), default="lfw")
    main(ap.parse_args().dataset)
