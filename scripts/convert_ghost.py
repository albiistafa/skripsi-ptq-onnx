"""GhostFaceNet .h5 -> ONNX FP32 (bersih). NHWC (1,112,112,3).

Jalankan dgn venv_tf (butuh tensorflow + tf2onnx).
"""
import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
import tf2onnx
import onnx
from onnx import TensorProto, numpy_helper

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
H5 = os.path.join(ROOT, "base_model", "GN_W1.3_S1_ArcFace_epoch46.h5")
RAW = os.path.join(ROOT, "onnx", "ghost_fp32_raw.onnx")
OUT = os.path.join(ROOT, "onnx", "ghost_fp32.onnx")

print("Memuat:", H5)
m = tf.keras.models.load_model(H5, compile=False)
print("  input", m.input_shape, "output", m.output_shape)

spec = (tf.TensorSpec((None, 112, 112, 3), tf.float32, name="input_face"),)
tf2onnx.convert.from_keras(m, input_signature=spec, opset=13, output_path=RAW)
print("raw ONNX:", RAW)

# Bersihkan float16 -> float32 (model GhostFaceNet mixed_float16).
model = onnx.load(RAW)
for n in model.graph.node:
    if n.op_type == "Cast":
        for a in n.attribute:
            if a.name == "to" and a.i == TensorProto.FLOAT16:
                a.i = TensorProto.FLOAT
for i, init in enumerate(model.graph.initializer):
    if init.data_type == TensorProto.FLOAT16:
        arr = numpy_helper.to_array(init).astype(np.float32)
        model.graph.initializer[i].CopyFrom(numpy_helper.from_array(arr, init.name))
onnx.checker.check_model(model)
onnx.save(model, OUT)
print("FP32 bersih:", OUT, f"({os.path.getsize(OUT)/1e6:.2f} MB)")
