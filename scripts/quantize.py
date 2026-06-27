"""Terapkan M1/M2/M3 PTQ via ONNX Runtime ke kedua model (identik).

M1 Dynamic INT8 | M2 Float16 | M3 Static INT8 QDQ (kalibrasi 200 citra).
Status tiap konversi -> results/compatibility.csv (Tabel A).
"""
import csv
import os
import numpy as np
import onnx
from onnxconverter_common import float16
from onnxruntime.quantization import (quantize_dynamic, quantize_static,
                                      QuantType, QuantFormat,
                                      CalibrationDataReader)
from onnxruntime.quantization.shape_inference import quant_pre_process

from common import ONNX_DIR, DATA_DIR, RESULTS_DIR, MODELS

os.makedirs(RESULTS_DIR, exist_ok=True)


class Reader(CalibrationDataReader):
    def __init__(self, npy, input_name):
        self.data = np.load(npy).astype("float32")
        self.input_name = input_name
        self.i = 0

    def get_next(self):
        if self.i >= len(self.data):
            return None
        x = self.data[self.i:self.i + 1]
        self.i += 1
        return {self.input_name: x}

    def rewind(self):
        self.i = 0


def m1_dynamic(fp32, out, inp):
    # QUInt8 (bukan QInt8): ConvInteger di ORT CPU butuh bobot unsigned, jika
    # tidak inferensi gagal (NOT_IMPLEMENTED) pada model padat-Conv.
    quantize_dynamic(fp32, out, weight_type=QuantType.QUInt8)


def m2_float16(fp32, out, inp):
    onnx.save(float16.convert_float_to_float16(onnx.load(fp32)), out)


def m3_static(fp32, out, inp):
    calib = os.path.join(DATA_DIR,
                         f"calib_{'nchw' if inp == 'input' else 'nhwc'}.npy")
    pre = fp32.replace(".onnx", "_pre.onnx")
    quant_pre_process(fp32, pre)  # shape-inference agar QDQ rapi
    quantize_static(pre, out, calibration_data_reader=Reader(calib, inp),
                    quant_format=QuantFormat.QDQ, per_channel=True,
                    weight_type=QuantType.QInt8,
                    activation_type=QuantType.QInt8)
    os.remove(pre)


TECHNIQUES = [("M1_dyn", m1_dynamic), ("M2_fp16", m2_float16),
              ("M3_qdq", m3_static)]


def main():
    rows = []
    for mname, cfg in MODELS.items():
        fp32 = os.path.join(ONNX_DIR, f"{mname}_fp32.onnx")
        inp = cfg["input"]
        for code, fn in TECHNIQUES:
            out = os.path.join(ONNX_DIR, f"{mname}_{code}.onnx")
            try:
                fn(fp32, out, inp)
                mb = os.path.getsize(out) / 1e6
                print(f"  OK  {mname:5s} {code:8s} -> {mb:.2f} MB")
                rows.append((mname, code, "OK", f"{mb:.2f}", ""))
            except Exception as e:  # noqa: BLE001
                print(f"  GAGAL {mname:5s} {code:8s}: {e}")
                rows.append((mname, code, "GAGAL", "", str(e)[:200]))

    with open(os.path.join(RESULTS_DIR, "compatibility.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "teknik", "status", "ukuran_MB", "pesan"])
        w.writerows(rows)
    print("\n-> results/compatibility.csv")


if __name__ == "__main__":
    main()
