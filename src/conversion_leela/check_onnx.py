import onnx
import torch

# infer random 1024,8,8,19 tensor
input = torch.randn(1024,112,8,8)
import onnxruntime as ort
# Load the model and create InferenceSession
model_path = "leela.onnx"
session = ort.InferenceSession(model_path)

# Run inference
outputs = session.run( None, {"x": input.numpy()})
print(outputs[0], outputs[1], outputs[2])