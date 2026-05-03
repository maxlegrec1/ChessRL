import chess
import random
import numpy as np
from src.mcts import SmallMCTS

def generate_random_fens(n=100):
    fens = []
    board = chess.Board()
    for _ in range(n):
        # Make some random moves to get a random position
        board.reset()
        for _ in range(random.randint(10, 40)):
            if board.is_game_over():
                break
            moves = list(board.legal_moves)
            board.push(random.choice(moves))
        fens.append(board.fen())
    return fens

def main():
    print("Initializing MCTS with model_dynamic_initial.trt...")
    try:
        mcts = SmallMCTS(engine_path="model_dynamic_initial.trt")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print("Checking specific positions...")
    
    start_fen = chess.STARTING_FEN
    # e4
    e4_fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    
    fens = [start_fen, e4_fen]
    results = mcts.run(fens, simulations=0)
    
    for fen, res in zip(fens, results):
        board = chess.Board(fen)
        val = res['value']
        print(f"\nFEN: {fen}")
        print(f"Turn: {'White' if board.turn == chess.WHITE else 'Black'}")
        print(f"Oriented Value (Side-to-move): Win={val[0]:.4f}, Draw={val[1]:.4f}, Loss={val[2]:.4f}")
        
        if board.turn == chess.BLACK:
            # Raw = Value
            raw_w = val[0]
            raw_l = val[2]
        else:
            # Raw = Flipped Value
            raw_w = val[2]
            raw_l = val[0]
            
        print(f"Raw (Black Persp): W_b={raw_w:.4f}, L_b={raw_l:.4f}")
        print(f"Bias (L_b - W_b): {raw_l - raw_w:.4f}")

    # Also check the random fens again but print the first few
    print("\nChecking random fens (first 5)...")
    print("Running inference (100 simulations)...")
    results = mcts.run(fens, simulations=100)
    for fen, res in zip(fens, results):
        val = res['value']
        board = chess.Board(fen)
        print(f"FEN: {fen} ({'W' if board.turn else 'B'}) -> Win={val[0]:.4f}")

if __name__ == "__main__":
    main()
