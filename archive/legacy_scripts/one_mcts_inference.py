from src.mcts import SmallMCTS

i = 41900
engine_path = f"engines/step_000{i}.trt"
small = SmallMCTS(engine_path=engine_path, cpuct=1.0,temperature = 1)

fen = "rn1qkbnr/ppp1pppp/3p4/8/3PP1Q1/8/PPP2PPP/RNB1KBNR b KQkq - 0 3"
res = small.run([fen], temperature=1, simulations=800,temperature_at_root=0.01, cpuct=1.0, use_dirichlet=False)
print("MCTS Result:", res[0])

from src.models.small.small_model import SmallModel
import torch
from src.models.small.vocab import policy_index
model = SmallModel()
model.load_state_dict(torch.load(f"checkpoints/step_000{i}.pt",weights_only=False)["model"])
model.eval()
model.to("cuda")
value = model.get_position_value(fen, device="cuda")
print("Value:", value)
move, probs, pre_probs = model.get_move_from_fen_no_thinking(
    fen, T=1, device="cuda", return_probs=True, return_pre_probs=True
)

# print the probabilities of the highest 20 moves, showing before/after legal mask
probs = probs.squeeze(0)  # remove batch dim
pre_probs = pre_probs.squeeze(0)
header = f"{'Move':8} {'Pre':>10} {'Post':>10} {'Delta':>10}"
print("\nTop 40 policy probs (before vs after legal mask):")
print(header)
print("-" * len(header))

rows = []
for pre, post, move_name in zip(pre_probs.tolist(), probs.tolist(), policy_index):
    rows.append((post, pre, move_name))

for post, pre, move_name in sorted(rows, key=lambda x: x[0], reverse=True)[:40]:
    delta = post - pre
    print(f"{move_name:8} {pre:10.4f} {post:10.4f} {delta:10.4f}")

