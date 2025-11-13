# TensorRT policy head examples
import sys
import time
from pathlib import Path

from src.models.leela.inference_cpp import Leelacpp
from src.models.small.inference_cpp import CppPolicy

ROOT = Path(__file__).resolve().parent
CPP_DIR = ROOT / "cpp"
if str(CPP_DIR) not in sys.path:
    sys.path.insert(0, str(CPP_DIR))

from src.mcts import AdversarialMCTS, LeelaMCTS, SmallMCTS


def run_policy_examples() -> None:
    fen = "r1bqkb1r/p4ppp/2p2n2/nB2p1N1/8/5Q2/PPPP1PPP/RNB1K2R b KQkq - 1 8"
    cpp_policy = CppPolicy(engine_path="model.trt")
    start_time = time.time()
    moves, wdl = cpp_policy([fen] * 1024, temperature=0.1, iterations=1)
    end_time = time.time()
    print(f"Time taken to run inference: {end_time - start_time:.2f} seconds")
    print(moves[:8], wdl[:8])

    cpp_policy = Leelacpp(engine_path="leela.trt")
    move_sequence = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "g8f6", "e1g1"]
    move_lists = [move_sequence] * 1024
    origin_fens = [""] * len(move_lists)
    cpp_moves, value_winner, value_q = cpp_policy(origin_fens, move_lists, temperature=0.1)
    print("TensorRT moves (custom FEN):", cpp_moves[:1])
    print("TensorRT value_winner (custom FEN):", value_winner[:1])
    print("TensorRT value_q (custom FEN):", value_q[:1])


def run_mcts_examples() -> None:
    fen = "r1bqkb1r/p4ppp/2p2n2/nB2p1N1/8/5Q2/PPPP1PPP/RNB1K2R b KQkq - 1 8"

    small = SmallMCTS(engine_path="model_minibatch.trt", simulations=256, cpuct=1.2)
    small_result = small.run(fen, temperature=0.8)
    print("Small MCTS action:", small_result["action"])
    print("Small MCTS top policy entries:", list(small_result["policy"].items())[:5])
    print("Small MCTS q_values sample:", list(small_result["q_values"].items())[:5])

    fen = ""
    history = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "g8f6", "e1g1"]

    leela = LeelaMCTS(engine_path="leela_minibatch.trt", simulations=256, cpuct=1.2)
    leela_result = leela.run(fen, history, temperature=0.6)
    print("Leela MCTS action:", leela_result["action"])
    print("Leela MCTS top policy entries:", list(leela_result["policy"].items())[:5])
    print("Leela MCTS q_values sample:", list(leela_result["q_values"].items())[:5])

    adversarial = AdversarialMCTS(
        attacker_engine_path="model_minibatch.trt",
        defender_engine_path="leela_minibatch.trt",
        simulations=256,
        cpuct=1.2,
    )
    adversarial_result = adversarial.run(fen, history, temperature=0.7)
    print("Adversarial MCTS action:", adversarial_result["action"])
    print("Adversarial MCTS top policy entries:", list(adversarial_result["policy"].items())[:5])
    print("Adversarial MCTS q_values sample:", list(adversarial_result["q_values"].items())[:5])


if __name__ == "__main__":
    run_policy_examples()
    run_mcts_examples()