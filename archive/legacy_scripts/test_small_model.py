from src.models.small.inference import Inference
import torch
model = Inference()
model.load_state_dict(torch.load("src/models/small/model.pt"), strict=False)
model.eval()
model.to("cuda")

fen = "8/5P2/kp2R3/4Q3/p1p5/2N3p1/PP4P1/5RK1 w - - 6 34"
value = model.get_position_value(fen, device="cuda")
print(value)