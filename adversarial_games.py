

from src.mcts import AdversarialMCTS, SmallMCTS, LeelaMCTS
from src.models.leela import BT4 as LeelaBT4
from src.generate.openings import hf_opening_fen_generator, parquet_opening_fen_generator
import torch
import bulletchess
from bulletchess import CHECKMATE, DRAW
import chess
from chess import pgn
from datetime import datetime
from tqdm import tqdm
import time
import json
import os
import pyarrow as pa
import pyarrow.parquet as pq

adversarial = AdversarialMCTS(attacker_engine_path="model_minibatch_adversarial.trt", defender_engine_path="leela_minibatch.trt", temperature=1, cpuct=1.0)
# adversarial = SmallMCTS(engine_path="model_minibatch.trt", temperature=1, cpuct=1.0)
# adversarial = LeelaMCTS(engine_path="leela_minibatch.trt", temperature=1, cpuct=1.0)

leela = LeelaBT4().to("cuda")
leela.load_state_dict(torch.load("src/models/leela/model.pt"),strict=False)

total_games = 1000
score = 0.0
num_games = 0
pgn_filename = "benchmark.pgn"
# Progress bar showing win rate
pbar = tqdm(range(total_games), desc="Adversarial Games")

# Open PGN file for writing games successively
pgn_file = open(pgn_filename, "w", encoding="utf-8")

# Prepare Parquet writer for incremental logging of adversarial decisions
parquet_filename = "adversarial_moves.parquet"
parquet_schema = pa.schema(
    [
        ("fen", pa.string()),
        ("move", pa.string()),
        ("value_json", pa.string()),
        ("q", pa.float32()),
    ]
)
# Overwrite per run; append within this run by reusing the writer
parquet_writer = pq.ParquetWriter(parquet_filename, parquet_schema)

# openings_gen = hf_opening_fen_generator(split="train", seed=42)
openings_gen = parquet_opening_fen_generator(parquet_path="src/generate/openings.parquet", seed=42)

STOCKFISH_PATH = "/mnt/2tb/ChessRL/stockfish/stockfish-ubuntu-x86-64-avx2"  # Adjust path if needed
STOCKFISH_ELO = 2900  # Set the desired ELO rating
engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
engine.configure({"UCI_LimitStrength": True, "UCI_Elo": STOCKFISH_ELO})
            
def get_stockfish_move(engine_obj, bullet_board):
    # Use python-chess board for Stockfish, then convert back to bulletchess.Move
    ch_board = chess.Board(bullet_board.fen())
    result = engine_obj.play(ch_board, chess.engine.Limit(time=0.1))
    return result.move.uci()


try:
    for _ in pbar:
        opening = next(openings_gen)
        history = opening["uci"]
        board = bulletchess.Board()
        for move in history:
            board.apply(bulletchess.Move.from_uci(move))

        adversarial_side = board.turn

        pgn_game = pgn.Game()
        pgn_game.headers["Event"] = "Adversarial Game"
        pgn_game.headers["Site"] = "Local Machine"
        pgn_game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
        pgn_game.headers["Round"] = "1"
        # pgn_game.headers["White"] = "Adversarial"
        # pgn_game.headers["Black"] = "Leela"
        if adversarial_side == bulletchess.WHITE:
            pgn_game.headers["White"] = "Adversarial"
            pgn_game.headers["Black"] = "Leela"
        else:
            pgn_game.headers["White"] = "Leela"
            pgn_game.headers["Black"] = "Adversarial"
        pgn_game.headers["FEN"] = board.fen()

        node = pgn_game
        while not (board in CHECKMATE or board in DRAW):
            if board.turn == adversarial_side:
                res = adversarial.run("", history, temperature=1, simulations=50, cpuct=1.0)
                move = res["action"]
                # Capture FEN and MCTS outputs for Parquet logging (adversarial side only)
                fen_before = board.fen()
                mcts_value = res.get("value", None)
                chosen_q = None
                qmap = res.get("q_values", {})
                if isinstance(qmap, dict):
                    chosen_q = qmap.get(move, None)
                # Write a single-row table incrementally
                try:
                    row = {
                        "fen": [fen_before],
                        "move": [move],
                        "value_json": [json.dumps(mcts_value) if mcts_value is not None else None],
                        "q": [None if chosen_q is None else float(chosen_q)],
                    }
                    parquet_writer.write_table(pa.Table.from_pydict(row, schema=parquet_schema))
                except Exception as _e:
                    print(f"Failed to write to Parquet: {_e}")
                    pass
                # move = list(res["policy"])[0]
                # try:
                #     assert move ==  list(res["policy"])[0] 
                # except AssertionError:
                #     print(f"Move mismatch: {move} != {list(res['policy'])[0]}")
                #     print(f"Policy: {res['policy']}")
                #     print(f"Action: {res['action']}")
                #     print(f"History: {history}")
                #     print(f"Board: {board.fen()}")
                #     raise
                # moves = leela.batch_get_moves_from_move_lists([history], T=0.1, device="cuda")
                # move = moves[0]

            else:
                moves = leela.batch_get_moves_from_move_lists([history], T=0.1, device="cuda")
                move = moves[0]
                # move = get_stockfish_move(engine, board)

            pgn_move = chess.Move.from_uci(move)
            history.append(move)

            board.apply(bulletchess.Move.from_uci(move))
            node = node.add_variation(pgn_move)

        if board in CHECKMATE:
            if board.turn != adversarial_side:
                score += 1
                if adversarial_side == bulletchess.WHITE:
                    result = "1-0"
                else:
                    result = "0-1"
            else:
                score += 0
                if adversarial_side == bulletchess.WHITE:
                    result = "0-1"
                else:
                    result = "1-0"
        elif board in DRAW:
            score += 0.5
            result = "1/2-1/2"
        pgn_game.headers["Result"] = result
        # Write game immediately to file
        exporter = chess.pgn.FileExporter(pgn_file)
        pgn_game.accept(exporter)
        pgn_file.write("\n\n")
        pgn_file.flush()  # Ensure it's written to disk
        
        num_games += 1
        if num_games:
            pbar.set_postfix({"Win Rate": score / num_games})

    final_win_rate = score / num_games if num_games else 0.0
    print(f"Final Win Rate: {final_win_rate}")
finally:
    # Close the PGN file
    pgn_file.close()
    # Close the engine
    engine.quit()
    # Close Parquet writer
    try:
        parquet_writer.close()
    except Exception:
        pass