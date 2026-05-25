from src.mcts import SmallMCTS
from src.models.small.small_model import SmallModel
import torch

# 1. PyTorch Model Inference
print("--- PyTorch Model Inference ---")
model = SmallModel().to("cuda")
ckpt_path = "checkpoints/step_00005000.pt"

ckpt = torch.load(ckpt_path, map_location="cuda", weights_only=False)
model.load_state_dict(ckpt["model"])

engine_path = "model_dynamic_step_00005000.trt"
small = SmallMCTS(engine_path=engine_path, cpuct=1.0, simulations=32)

fen = "r2r4/pp1k2pp/2pP4/Q4pN1/2P3P1/1B5b/PP1R3P/RNB1K3 b Q - 2 27"
res = small.run([fen], temperature=1, simulations=32,temperature_at_root=0.5, cpuct=1.0, use_dirichlet=False)
print("MCTS Result:", res[0])

value_start = model.get_position_value(fen,device="cuda")
print("Value in worse for black position", value_start)

fen = "r1kr4/pp4pp/2pP4/Q4pN1/2P3P1/1B5b/PP1R3P/RNB1K3 w Q - 3 28"

value = model.get_position_value(fen, device="cuda")
print("Value in mate in one position:", value, [value[0]- value[2]])

fen = "r3r3/pp1k2pp/2pP4/Q4pN1/2P3P1/1B5b/PP1R3P/RNB1K3 w Q - 3 28"
value = model.get_position_value(fen, device="cuda")
print("Value in better move by black:", value, [value[0]- value[2]])


fen = "r2r4/pp1k2pp/2pP4/Q4pN1/2P3P1/1B5b/PP1R3P/RNB1K3 b Q - 2 27"
res = small.run([fen], temperature=1, simulations=800,temperature_at_root=0.5, cpuct=1.0, use_dirichlet=False)
print("MCTS Result:", res[0])

engine_path = "model_dynamic_initial.trt"
small = SmallMCTS(engine_path=engine_path, cpuct=1.0, simulations=32)
res = small.run([fen], temperature=1, simulations=800,temperature_at_root=0.5, cpuct=1.0, use_dirichlet=False)
print("MCTS Result:", res[0])