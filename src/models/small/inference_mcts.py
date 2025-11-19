from src.mcts import SmallMCTS

class SmallMCTSInference():
    def __init__(self, engine_path: str = "model_minibatch.trt",cpuct: float = 1.0, simulations: int = 100):
        self.simulations = simulations
        self.cpuct = cpuct
        self.mcts = SmallMCTS(engine_path=engine_path, temperature=1, cpuct=cpuct)
    def get_move_from_fen_no_thinking(self, fen, T = 1,device = "cuda",force_legal = True,return_probs = False, use_fp16: bool = False):
        res = self.mcts.run(fen, temperature=T, simulations=self.simulations,cpuct=self.cpuct)
        action = res["action"]
        return action

if __name__ == "__main__":
    inference = SmallMCTSInference(engine_path="model_minibatch.trt",cpuct=1.0, simulations=100)
    fen = "r1bqkb1r/p4ppp/2p2n2/nB2p1N1/8/5Q2/PPPP1PPP/RNB1K2R b KQkq - 1 8"
    action = inference.get_move_from_fen_no_thinking(fen, T=0.0)
    print(action)