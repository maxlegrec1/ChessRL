from src.mcts import SmallMCTS

small = SmallMCTS(engine_path="model_minibatch_adversarial.trt", temperature=1, cpuct=1.0)
fen = "5k2/8/3K4/6Q1/8/8/8/8 w - - 11 64"
fen = "6k1/p4ppp/1p6/8/8/1N6/P1rr1PPP/4R1K1 w - - 0 1" # mate in one
fen = "6k1/p4ppp/1p6/8/8/1N6/P1r1rPPP/3R2K1 w - - 0 1" # mate in two
fen = "8/8/6K1/3k4/8/6P1/1r6/4q3 b - - 1 60" # mate in four
fen = "r1b2r1k/1pqnbppp/p1n1p3/3pP3/1P3P2/P1N1BN2/2PQB1PP/R4RK1 w - - 5 14"
fen = "rn3rk1/p7/bpq1pp1Q/3pP3/P2Nn3/1Pb2NPP/5PB1/3RR1K1 w - - 1 22" # mate in eight
fen = "r2qkb1r/1p2pppp/p1n1b3/2nN4/2B5/5N2/PPP2PPP/R1BQK2R w KQkq - 0 9"
res = small.run(fen, temperature=1, simulations=1, cpuct=1.0)
print(res)

# import torch
# from src.models.small.inference import Inference
# model = Inference().to("cuda")
# model.load_state_dict(torch.load("src/models/small/model.pt"),strict=False)
# fen = "6k1/6p1/2p4r/p2p4/1p4p1/1BP2b2/Pq1B4/4QRK1 w - - 2 41"
# value = model.get_position_value(fen, device="cuda")
# print(value)

# # import torch
# from src.mcts import LeelaMCTS
# leela = LeelaMCTS(engine_path="leela_minibatch.trt", temperature=1, cpuct=1.0)


# fen = "6k1/6p1/2p1r1q1/p2p4/1p4p1/1BP2bQ1/PP1B3R/6K1 w - - 6 36"
# history = ["h2f2","e6f6","d2c1","g6e4","c1d2","e4b1","f2f1","b1b2","g3e1"]

# # # fen = ""
# # # history = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5","g8f6","e1g1"]

# res = leela.run(fen,history, temperature=1, simulations=1000, cpuct=1.0)
# print(res)
# from src.models.leela import BT4 as LeelaBT4
# leela_bt4 = LeelaBT4().to("cuda")
# leela_bt4.load_state_dict(torch.load("src/models/leela/model.pt"),strict=False)

# fen = "6k1/6p1/2p1r1q1/p2p4/1p4p1/1BP2bQ1/PP1B3R/6K1 w - - 6 36"
# history = ["h2f2","e6f6","d2c1","g6e4","c1d2","e4b1","f2f1","b1b2","g3e1"]

# moves = leela_bt4.batch_get_moves_from_move_lists([history], T=0.1, device="cuda", fens=[fen], use_fp16=True)
# print(moves)

# import time
# from src.mcts import AdversarialMCTS
# adversarial = AdversarialMCTS(attacker_engine_path="model_minibatch.trt", defender_engine_path="leela_minibatch.trt", temperature=1, cpuct=1.0)


# fen = "6k1/6p1/2p1r1q1/p2p4/1p4p1/1BP2bQ1/PP1B3R/6K1 w - - 6 36"
# history = ["h2f2","e6f6","d2c1","g6e4","c1d2","e4b1","f2f1","b1b2","g3e1"]

# res = adversarial.run(fen,history, temperature=1, simulations=10000, cpuct=1.0)
# print(res)