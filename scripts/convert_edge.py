"""EdgeFace (PyTorch) -> ONNX FP32. NCHW (1,3,112,112)."""
import os, torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "onnx", "edge_fp32.onnx")
BASE = os.path.join(ROOT, "base_model", "edgeface_s_gamma_05.pt")

print("Mengunduh EdgeFace via torch.hub ...")
model = torch.hub.load("otroshi/edgeface", "edgeface_s_gamma_05",
                       source="github", pretrained=True)
model.eval()

# simpan bobot base model PyTorch ke base_model/
torch.save(model.state_dict(), BASE)
print("base model tersimpan:", BASE, f"({os.path.getsize(BASE)/1e6:.2f} MB)")

dummy = torch.randn(1, 3, 112, 112)
with torch.no_grad():
    emb = model(dummy)
print("output shape:", tuple(emb.shape))

torch.onnx.export(model, dummy, OUT,
                  input_names=["input"], output_names=["emb"],
                  opset_version=13,
                  dynamic_axes={"input": {0: "N"}, "emb": {0: "N"}})
print("tersimpan:", OUT, f"({os.path.getsize(OUT)/1e6:.2f} MB)")
