
from src.models.small.inference import Inference
import torch

model = Inference().to("cuda")


model.load_state_dict(torch.load("checkpoints/checkpoint_step_2000.pt",weights_only=False)["model_state_dict"],strict=False)


torch_model = model
# Create example inputs for exporting the model. The inputs should be a tuple of tensors.
example_inputs = torch.randn(1024, 8, 8, 19,device='cuda')

onnx_program = torch.onnx.export(torch_model, example_inputs, dynamo=True)
onnx_program.save("model_.onnx")

