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

fen = "r3k2r/ppp2ppp/2P5/8/2p1q1b1/B1P2N2/P4PPP/R2Q2K1 b kq - 0 14"

start_time = time.time()
moves = model.batch_get_moves_from_fens([fen]*1024, T=0.1, use_fp16 = True)
end_time = time.time()
print(f"Time taken to run inference: {end_time - start_time:.2f} seconds")
print(moves)

from src.models.small.inference_cpp import CppPolicy
cpp_policy = CppPolicy()
start_time = time.time()
moves = cpp_policy([fen]*1024, temperature=0.1, iterations=1)
end_time = time.time()
print(f"Time taken to run inference: {end_time - start_time:.2f} seconds")