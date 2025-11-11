import bulletchess
from src.models.small import BT4

pgn_path = "runs/pgn/game_1762696726860_864.pgn"

with bulletchess.pgn.PGNFile.open(pgn_path) as f:
    game = f.next_game()
    opening_uci_str = game["OpeningUCI"]
    print(game["White"])
    opening_moves = opening_uci_str.split()
    print(opening_moves)
    len_opening = len(opening_moves)
    print(game.starting_board.fen())