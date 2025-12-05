import torch
import onnx
from src.models.small.small_model import SmallModel

# 1. Load Model
ckpt_path = "checkpoints/step_00005000.pt"
model = SmallModel().to("cuda")
# model = Inference(num_layers=10, d_model=512, d_ff=1024, num_heads=8).to("cuda")
# checkpoint = torch.load("src/models/small/model.pt", map_location="cuda", weights_only=False)
ckpt = torch.load(ckpt_path, map_location="cuda", weights_only=False)
model.load_state_dict(ckpt["model"])
print(f"Loaded {ckpt_path}")
model.eval()

class ExportWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        p, v, vq = self.model(x, return_all=True)
        v_probs = torch.softmax(v, dim=-1)
        return p, v_probs

model = ExportWrapper(model).to("cuda")

# 2. Define Dynamic Axes (Works perfectly with Legacy Exporter)
dynamic_axes = {
    'input': {0: 'batch_size'},
    'policy': {0: 'batch_size'},
    'value': {0: 'batch_size'}
}

# 3. Create Dummy Input
dummy_input = torch.randn(1, 8, 8, 19, device="cuda:0")

# 4. Export
output_file = "model_dynamic_step_00005000.onnx"
print(f"Exporting to {output_file} using Legacy Exporter (Opset 17)...")

# CRITICAL: Do NOT pass dynamo=True. 
# The default behavior respects dynamic_axes.
torch.onnx.export(
    model,
    dummy_input,
    output_file,
    input_names=['input'],
    output_names=['policy', 'value'],
    dynamic_axes=dynamic_axes,
    opset_version=17,
    export_params=True,
    do_constant_folding=True,
    dynamo=False
)

# 5. Verify
print(f"\nVerifying {output_file}...")
model_onnx = onnx.load(output_file)
input_shape = model_onnx.graph.input[0].type.tensor_type.shape.dim

# Robustly extract the dimension value or param string
dim0 = "?"
if len(input_shape) > 0:
    if input_shape[0].HasField("dim_param"):
        dim0 = input_shape[0].dim_param
    elif input_shape[0].HasField("dim_value"):
        dim0 = str(input_shape[0].dim_value)

print(f"Input Dimension 0 is: '{dim0}'")

if dim0 == "batch_size":
    print("✅ SUCCESS: Model is dynamic.")
else:
    print(f"❌ FAILURE: Model is static (Batch size fixed to {dim0}).")