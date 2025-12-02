from src.mcts import SmallMCTS
import torch
import numpy as np
from src.models.small.small_model import SmallModel
from src.models.small.fen_encoder import fen_to_tensor
from src.models.small.vocab import policy_index

# 1. PyTorch Model Inference
print("--- PyTorch Model Inference ---")
model = SmallModel().to("cuda")
ckpt_path = "checkpoints/step_00319850.pt"

ckpt = torch.load(ckpt_path, map_location="cuda", weights_only=False)
model.load_state_dict(ckpt["model"])
print(f"Loaded {ckpt_path}")
model.eval()

fen = "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2"

# # Use the new method
# move, probs = model.get_move_from_fen_no_thinking(fen, T=0.0, device="cuda", return_probs=True)
# print(f"Selected move: {move}")

value = model.get_position_value(fen, device="cuda")
print(f"Value (WDL): {value}")

# # Top 10 moves
# print("Top 10 Policy Moves (Masked):")
# sorted_probs, indices = torch.sort(probs, dim=-1, descending=True)
# for i in range(10):
#     idx = indices[0][i].item()
#     prob = sorted_probs[0][i].item()
#     if prob > 0:
#         print(f"{policy_index[idx]}: {prob:.4f}")

# # # 2. MCTS Inference
# print("\n--- MCTS Inference (TensorRT) ---")
# small = SmallMCTS(engine_path="model_dynamic_good.trt", temperature=1, cpuct=1.0)
# res = small.run([fen], temperature=1, simulations=800, cpuct=1.0, temperature_at_root=1.4, use_dirichlet=True, dirichlet_alpha=0.3, dirichlet_epsilon=0.25)
# print("MCTS Result:", res[0])

from mate_in_one_fens import MATE_IN_ONE_FENS

avg_sdm_value = 0

acc = 0


for fen in MATE_IN_ONE_FENS:
    value = model.get_position_value(fen, device="cuda")
    side = fen.split(" ")[1]
    print(f"Fen: {fen}\nSide: {side}\nValue (WDL): {value}")
    if side == "w":
        if value[2] > value[0]:
            acc += 1
        value = value[2]

    else:
        if value[0] > value[2]:
            acc += 1
        value = value[0]
    avg_sdm_value += value

print(f"Average SDM Value: {avg_sdm_value / len(MATE_IN_ONE_FENS)}")

print(f"Accuracy: {acc / len(MATE_IN_ONE_FENS)}")