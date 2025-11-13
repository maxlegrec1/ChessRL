
from src.models.small.inference import Inference
import torch

model = Inference().to("cuda")


# model.load_state_dict(torch.load("checkpoints/checkpoint_step_2000.pt",weights_only=False)["model_state_dict"],strict=False)
model.load_state_dict(torch.load("src/models/small/model.pt",weights_only=False),strict=False)


# fen = "r3k2r/ppp2ppp/2P5/8/2p1q1b1/B1P2N2/P4PPP/R2Q2K1 b kq - 0 14"

# move = model.get_move_from_fen_no_thinking(fen, T=0.0, use_fp16 = True)
# moves = model.batch_get_moves_from_fens([fen]*1024, T=0.1, use_fp16 = True)
# value = model.get_position_value(fen)

# print(f"Move played from FEN: {move}")
# print(f"Moves played from FEN: {moves}")
# print(f"Value of FEN: {value}")


torch_model = model
# Create example inputs for exporting the model. The inputs should be a tuple of tensors.
example_inputs = torch.randn(32, 8, 8, 19,device='cuda')

onnx_program = torch.onnx.export(torch_model, example_inputs, dynamo=True)
onnx_program.save("model_minibatch.onnx")

