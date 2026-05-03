from typing import Sequence

from src.mcts import LeelaMCTS


class LeelaMCTSInference:
    def __init__(
        self,
        engine_path: str = "leela_minibatch.trt",
        cpuct: float = 1.0,
        simulations: int = 800,
    ):
        self.simulations = simulations
        self.cpuct = cpuct
        self.mcts = LeelaMCTS(engine_path=engine_path, temperature=1, cpuct=cpuct)

    def get_move_from_fen_no_thinking(
        self,
        fen: str,
        T: float = 1,
        device: str = "cuda",
        force_legal: bool = True,
        return_probs: bool = False,
        use_fp16: bool = False,
        history: Sequence[str] = (),
    ) -> str:
        # Leela MCTS expects empty FEN when history is provided
        # It reconstructs the position from startpos + history
        effective_fen = "" if history else fen
        res = self.mcts.run(
            effective_fen,
            history=history,
            temperature=T,
            simulations=self.simulations,
            cpuct=self.cpuct,
        )
        action = res["action"]
        return action


if __name__ == "__main__":
    inference = LeelaMCTSInference(
        engine_path="leela_minibatch.trt", cpuct=1.0, simulations=100
    )
    fen = ""
    history = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "g8f6", "e1g1"]
    action = inference.get_move_from_fen_no_thinking(fen, T=1.0, history=history)
    print(action)
