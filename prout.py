from new_paradigm.start_v4 import ReasoningTransformer
from new_paradigm.model_v3_raw import BT4
import torch
import torch.nn as nn
from metrics.tactics_evaluation_improved import calculate_tactics_accuracy
from metrics.endgame_evaluation import calculate_endgame_score
from metrics.play_move_gemini import main_model_vs_stockfish
model = ReasoningTransformer()
model.load_state_dict(torch.load("fine_tune/V4_150000.pt"))
model.to("cuda")
#print number of parameters

fen = "r1bqkbnr/ppp2ppp/2np4/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 4"


move,cot = model.get_move_from_fen(fen,T=0.2,return_cot=True)
print(move,cot)
