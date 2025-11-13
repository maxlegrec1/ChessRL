from src.models.leela import BT4
import torch

model = BT4().to("cuda")

model.load_state_dict(torch.load("src/models/leela/model.pt",weights_only=False),strict=False)


torch_model = model
# Create example inputs for exporting the model. The inputs should be a tuple of tensors.
example_inputs = torch.randn(32, 112, 8, 8,device='cuda')

onnx_program = torch.onnx.export(torch_model, example_inputs, dynamo=True)
onnx_program.save("leela_minibatch.onnx")

