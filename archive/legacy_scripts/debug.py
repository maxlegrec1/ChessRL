from src.mcts import SmallMCTS
import torch
import numpy as np
from src.models.small.small_model import SmallModel
from src.models.small.fen_encoder import fen_to_tensor
from src.models.small.vocab import policy_index, policy_to_idx
import torch.nn.functional as F
import bulletchess

# 1. PyTorch Model Inference
print("--- PyTorch Model Inference ---")
model = SmallModel().to("cuda")
ckpt_path = "checkpoints/step_00005000.pt"

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
acc_white = 0
acc_black = 0
num_white = 0
num_black = 0

true_acc = 0
true_acc_white = 0
true_acc_black = 0

# Policy accuracy
policy_correct = 0
policy_correct_white = 0
policy_correct_black = 0

# Mating move probability tracking
mating_move_probs = []
mating_move_ranks = []

for fen, mating_move in MATE_IN_ONE_FENS:
    value = model.get_position_value(fen, device="cuda")
    side = fen.split(" ")[1]
    
    # Get model's move prediction
    predicted_move = model.get_move_from_fen_no_thinking(fen, T=0.0, device="cuda")
    is_correct = predicted_move == mating_move
    
    # Compute mating move probability and rank
    tensor = torch.from_numpy(fen_to_tensor(fen)).unsqueeze(0).to("cuda")
    with torch.no_grad():
        logits = model(tensor)
    
    board = bulletchess.Board.from_fen(fen)
    legal_moves = list(board.legal_moves())
    
    # Create mask for legal moves
    mask = torch.full((1, 1929), float('-inf'), device='cuda')
    for move in legal_moves:
        uci = move.uci()
        if uci.endswith('n'):
            uci = uci[:-1]
        if uci in policy_to_idx:
            mask[0, policy_to_idx[uci]] = 0.0
    
    masked_logits = logits + mask
    probs = F.softmax(masked_logits, dim=-1)
    
    # Get probability of mating move
    mating_uci = mating_move if not mating_move.endswith('n') else mating_move[:-1]
    if mating_uci in policy_to_idx:
        mating_prob = probs[0, policy_to_idx[mating_uci]].item()
    else:
        mating_prob = 0.0
    
    # Get rank of mating move
    move_probs = []
    for move in legal_moves:
        uci = move.uci()
        if uci.endswith('n'):
            uci = uci[:-1]
        if uci in policy_to_idx:
            prob = probs[0, policy_to_idx[uci]].item()
            move_probs.append((uci, prob))
    move_probs.sort(key=lambda x: -x[1])
    mating_rank = next((i+1 for i, (m, p) in enumerate(move_probs) if m == mating_uci), len(move_probs))
    
    mating_move_probs.append(mating_prob)
    mating_move_ranks.append(mating_rank)
    
    print(f"Fen: {fen}\nSide: {side}\nValue (WDL): {value}, {value[0]>value[2]}\nPredicted: {predicted_move}, Correct: {mating_move}, Match: {is_correct}\nMating move prob: {mating_prob*100:.2f}%, rank: {mating_rank}/{len(move_probs)}")
    
    # Policy accuracy
    if is_correct:
        policy_correct += 1
        if side == "w":
            policy_correct_white += 1
        else:
            policy_correct_black += 1
    
    if value[0] > value[2]:
        acc += 1
        if side == "w":
            acc_white += 1
        else:
            acc_black += 1
    if value[0] > value[2] and value[0] > value[1]:
        true_acc += 1
        if side == "w":
            true_acc_white += 1
        else:
            true_acc_black += 1
    if side == "w":
        num_white += 1
    else:
        num_black += 1
    value = value[0]
    avg_sdm_value += value

print(f"\n=== Value Accuracy ===")
print(f"Accuracy: {acc / len(MATE_IN_ONE_FENS)}")
print(f"Accuracy White: {acc_white / num_white}")
print(f"Accuracy Black: {acc_black / num_black}")
print(f"True Accuracy: {true_acc / len(MATE_IN_ONE_FENS)}")
print(f"True Accuracy White: {true_acc_white / num_white}")
print(f"True Accuracy Black: {true_acc_black / num_black}")

print(f"\n=== Policy Accuracy ===")
print(f"Policy Accuracy: {policy_correct / len(MATE_IN_ONE_FENS):.4f} ({policy_correct}/{len(MATE_IN_ONE_FENS)})")
print(f"Policy Accuracy White: {policy_correct_white / num_white:.4f} ({policy_correct_white}/{num_white})")
print(f"Policy Accuracy Black: {policy_correct_black / num_black:.4f} ({policy_correct_black}/{num_black})")

print(f"\n=== Mating Move Probability ===")
print(f"Average mating move probability: {np.mean(mating_move_probs)*100:.4f}%")
print(f"Average mating move rank: {np.mean(mating_move_ranks):.1f} / {len(legal_moves)}")
print(f"Median mating move probability: {np.median(mating_move_probs)*100:.4f}%")
print(f"Median mating move rank: {np.median(mating_move_ranks):.1f}")