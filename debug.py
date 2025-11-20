from src.mcts import SmallMCTS
import time
import matplotlib.pyplot as plt
import numpy as np

# Initialize MCTS
small = SmallMCTS(engine_path="model_dynamic_initial.trt", temperature=1, cpuct=1.0)
fen = "rnbqkbnr/pppppppp/8/8/P7/8/1PPPPPPP/RNBQKBNR b KQkq - 0 1"


res = small.run([fen] * 128, temperature=1, simulations=600, cpuct=1.0,temperature_at_root=0)

print([res["action"] for res in res])