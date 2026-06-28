"""Konverter mixed-precision fp16 manual yang BENAR.

Library standar (onnxconverter_common, ORT transformers) gagal utk EdgeFace:
mereka meng-cast SEMUA edge ke fp16 -> attention XCA (cross-covariance)
overflow (Div -> Inf -> NaN). Di sini node attention dipertahankan fp32 dan
edge di dalamnya TETAP fp32; hanya bagian Conv/MLP yang jadi fp16.

Algoritma:
  1. shape-infer -> tahu dtype tiap tensor (jangan sentuh tensor non-float).
  2. node precision: fp32 bila namanya cocok keep_fp32, selain itu fp16.
  3. initializer float -> fp16 bila SEMUA konsumennya fp16 (hemat ukuran).
  4. sisipkan Cast hanya di batas fp16<->fp32; I/O graph tetap fp32.
"""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

FP16_MAX = 65504.0


def _clip_fp16(arr):
    a = np.clip(arr, -FP16_MAX, FP16_MAX)
    return a.astype(np.float16)


def _float_tensors(model, feed):
    """Set nama tensor float32 dari SATU inferensi nyata (akurat, tak andalkan
    value_info yg sering tak lengkap utk Slice/Reshape/Constant)."""
    import onnxruntime as ort
    m = onnx.load_model_from_string(model.SerializeToString())
    base = {o.name for o in m.graph.output}
    for n in m.graph.node:
        for o in n.output:
            if o and o not in base:
                m.graph.output.append(helper.ValueInfoProto(name=o))
    s = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
    outs = s.run(None, feed)
    fset = {n for n, v in zip([o.name for o in s.get_outputs()], outs)
            if np.asarray(v).dtype == np.float32}
    fset |= {i.name for i in model.graph.initializer
             if i.data_type == TensorProto.FLOAT}
    fset |= {i.name for i in model.graph.input}  # input float
    return fset


def convert(model, keep_fp32_pred, feed):
    """keep_fp32_pred(node)->True bila node tetap fp32. feed: dict input fp32."""
    float_set = _float_tensors(model, feed)
    model = onnx.shape_inference.infer_shapes(model)
    g = model.graph
    is_float = lambda t: t in float_set

    prec = {n.name: ("fp32" if keep_fp32_pred(n) else "fp16") for n in g.node}
    produced_by = {o: n for n in g.node for o in n.output}
    graph_io = {v.name for v in list(g.input) + list(g.output)}

    consumers = {}
    for n in g.node:
        for t in n.input:
            consumers.setdefault(t, []).append(n)

    # 1) initializer float -> fp16 bila semua konsumen fp16.
    init_dtype = {}
    for init in g.initializer:
        if init.data_type != TensorProto.FLOAT:
            init_dtype[init.name] = init.data_type
            continue
        cons = consumers.get(init.name, [])
        if cons and all(prec[c.name] == "fp16" for c in cons):
            arr = _clip_fp16(numpy_helper.to_array(init))
            init.CopyFrom(numpy_helper.from_array(arr, init.name))
        init_dtype[init.name] = init.data_type

    # 1b) Constant node float -> fp16 bila node-nya fp16 (nilai literal ikut).
    for n in g.node:
        if n.op_type == "Constant" and prec[n.name] == "fp16":
            for a in n.attribute:
                if a.name == "value" and a.t.data_type == TensorProto.FLOAT:
                    arr = _clip_fp16(numpy_helper.to_array(a.t))
                    a.t.CopyFrom(numpy_helper.from_array(arr))

    def stored_dtype(t):
        if t in init_dtype:
            return "fp16" if init_dtype[t] == TensorProto.FLOAT16 else "fp32"
        if t in graph_io:
            return "fp32"            # I/O dipertahankan fp32
        p = produced_by.get(t)
        return prec[p.name] if p is not None else "fp32"

    # 2) sisipkan Cast di batas precision utk input tiap node.
    cast_cache = {}   # (tensor, target) -> nama tensor hasil cast
    new_nodes = []

    def cast_to(t, target):
        key = (t, target)
        if key in cast_cache:
            return cast_cache[key]
        out = f"{t}__to_{target}"
        to = TensorProto.FLOAT16 if target == "fp16" else TensorProto.FLOAT
        new_nodes.append(helper.make_node("Cast", [t], [out], to=to,
                                          name=f"Cast_{len(cast_cache)}_{target}"))
        cast_cache[key] = out
        return out

    for n in g.node:
        want = prec[n.name]
        for i, t in enumerate(n.input):
            if not t or not is_float(t):
                continue
            if stored_dtype(t) != want:
                n.input[i] = cast_to(t, want)

    # 3) graph output float yg diproduksi node fp16 -> cast balik ke fp32.
    for out in g.output:
        if not is_float(out.name):
            continue
        p = produced_by.get(out.name)
        if p is not None and prec[p.name] == "fp16":
            tmp = out.name + "__fp16"
            for nn in g.node:
                nn.output[:] = [tmp if o == out.name else o for o in nn.output]
            new_nodes.append(helper.make_node("Cast", [tmp], [out.name],
                                              to=TensorProto.FLOAT,
                                              name=f"Cast_out_{out.name}"))

    all_nodes = list(g.node) + new_nodes

    # urutkan topologis (Cast baru bisa berada di mana saja).
    available = {v.name for v in g.input} | {i.name for i in g.initializer} | {""}
    ordered, pending = [], list(all_nodes)
    while pending:
        progressed = False
        rest = []
        for n in pending:
            if all(t in available for t in n.input):
                ordered.append(n)
                available.update(n.output)
                progressed = True
            else:
                rest.append(n)
        pending = rest
        if not progressed:
            raise RuntimeError(f"siklus/again {len(pending)} node tak terurut")

    del g.node[:]
    g.node.extend(ordered)
    del g.value_info[:]            # biar ORT infer ulang tipe internal
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    return model
