from src.mcts import SmallMCTS
import time
import matplotlib.pyplot as plt
import numpy as np

# # Initialize MCTS
# small = SmallMCTS(engine_path="model_dynamic_initial.trt", temperature=1, cpuct=1.0)
# fen = "rnbqkbnr/pppppppp/8/8/P7/8/1PPPPPPP/RNBQKBNR b KQkq - 0 1"


# res = small.run([fen] * 128, temperature=1, simulations=600, cpuct=1.0,temperature_at_root=0,use_dirichlet=True,dirichlet_alpha=0.3,dirichlet_epsilon=0.25)

# print([res["action"] for res in res])

from src.models.small.model import BT4
import torch
model = BT4().to("cuda")
ckpt = torch.load("checkpoints/step_00010000.pt",weights_only=False)
model.load_state_dict(ckpt["model"])
model.eval()


fen = "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"

_,probs = model.get_move_from_fen_no_thinking(fen,1,device="cuda",return_probs=True)

value = model.get_position_value(fen,device="cuda")
print(value)



from src.models.small.model import policy_index
# order probs descending and print top 10 moves and their probabilities
sorted_probs, indices = torch.sort(probs,dim=-1,descending=True)

for i in range(10):
    print(f"{policy_index[indices[0][i].item()]}: {sorted_probs[0][i].item()}")


small = SmallMCTS(engine_path="engines/step_00010000.trt", temperature=1, cpuct=1.0)

res = small.run([fen] * 1, temperature=1, simulations=600, cpuct=1.0,temperature_at_root=1.4,use_dirichlet=True,dirichlet_alpha=0.3,dirichlet_epsilon=0.25)

print(res[0])