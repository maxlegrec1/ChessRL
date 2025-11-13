import torch
# from src.models.leela import BT4

# model = BT4()
# model.load_state_dict(torch.load("src/models/leela/model.pt"))
# model.to("cuda")
# # Move sequence to reach the same position
# # MOVE_SEQUENCE = ["g1h3",
# #     "d7d6",
# #     "f2f3",
# #     "c8g4",
# #     "h1g1",
# #     "h7h5",
# #     "g1h1",
# #     "a7a6",
# #     "d2d4",
# #     "g8h6",
# #     "d4d5"]

# MOVE_SEQUENCE = [
#     "d2d3"

# ]

# move = model.get_move_from_fen_no_thinking(MOVE_SEQUENCE, T=0.0, use_fp16 = True )
# moves = model.batch_get_moves_from_move_lists([MOVE_SEQUENCE]*1024, T=0.0, use_fp16 = True)
# print(f"Move played from FEN: {move}")
# print(f"Moves played from FEN: {moves}")
    
from src.models.small.inference import Inference
import time
model = Inference().to("cuda")
model.load_state_dict(torch.load("src/models/small/model.pt"), strict= False)
# model.load_state_dict(torch.load("checkpoints/checkpoint_step_2000.pt",weights_only=False)["model_state_dict"],strict=False)

fen = "r1bqkb1r/p4ppp/2p2n2/nB2p1N1/8/5Q2/PPPP1PPP/RNB1K2R b KQkq - 1 8"

# move, probs = model.get_move_from_fen_no_thinking(fen, T=1, use_fp16 = True,return_probs=True)
# print(move, probs)
# probs = probs.squeeze(0)
# # sort indices of probs by descending order
# sorted_indices = torch.argsort(probs, dim=-1, descending=True)
# print(sorted_indices)
# print(probs[sorted_indices.tolist()[:10]])
# from src.models.small.inference import policy_index
# # first 10 moves
# print([policy_index[i] for i in sorted_indices.tolist()[:10]])


# start_time = time.time()
# moves = model.batch_get_moves_from_fens([fen]*1024, T=0.1, use_fp16 = True)
# value = model.get_position_value(fen)
# end_time = time.time()
# print(f"Time taken to run inference: {end_time - start_time:.2f} seconds")
# print(value)

# from src.models.small.inference_cpp import CppPolicy
# cpp_policy = CppPolicy(engine_path="model.trt")
# start_time = time.time()
# moves,wdl = cpp_policy([fen]*1024, temperature=0.1, iterations=1)
# end_time = time.time()
# print(f"Time taken to run inference: {end_time - start_time:.2f} seconds")

# print(moves[:8], wdl[:8])

import bulletchess
from src.models.small.fen_encoder import fen_to_tensor
board = bulletchess.Board.from_fen(fen)
x = torch.from_numpy(fen_to_tensor(fen)).to("cuda").to(torch.float32)
x = x.view(1,8,8,19)

policy, _ = model(x)
print(policy.shape)
policy = policy.cpu().squeeze(0)
sorted_indices = torch.argsort(policy, dim=-1, descending=True)
sorted_indices = sorted_indices.squeeze(0)

print(sorted_indices)
print(policy[sorted_indices[:10]])
from src.models.small.inference import policy_index
print([policy_index[i] for i in sorted_indices[:10]])