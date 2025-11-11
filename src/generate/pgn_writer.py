from typing import List, Optional
import os

import chess
import chess.pgn


def write_pgn(moves_uci: List[str], out_path: str, white_name: str, black_name: str, result: str = "*", start_fen: str = chess.STARTING_FEN, opening_uci: Optional[List[str]] = None):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    board = chess.Board(fen=start_fen)
    for uci in moves_uci:
        mv = chess.Move.from_uci(uci)
        # If move not legal for this board (shouldn't happen), skip remaining
        if mv not in board.legal_moves:
            break
        board.push(mv)

    game = chess.pgn.Game.from_board(board)
    game.headers["White"] = white_name
    game.headers["Black"] = black_name
    game.headers["Result"] = result
    if start_fen != chess.STARTING_FEN:
        game.headers["FEN"] = start_fen
        game.headers["SetUp"] = "1"
    if opening_uci:
        try:
            game.headers["OpeningUCI"] = " ".join(opening_uci)
        except Exception:
            pass

    with open(out_path, "w", encoding="utf-8") as f:
        exporter = chess.pgn.FileExporter(f)
        game.accept(exporter)
