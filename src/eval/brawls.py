import argparse
import os
from datetime import datetime
from typing import Optional, Dict, Any, List

import torch
import chess  # python-chess, for PGN
import bulletchess
from chess import pgn
from bulletchess import CHECKMATE, DRAW


def _is_game_over(board: bulletchess.Board) -> bool:
    return (board in CHECKMATE) or (board in DRAW)


def _safe_get_move(
    model,
    board: bulletchess.Board,
    move_history: List[str],
    temp: float,
    device: str,
):
    """Return UCI move from model, supporting both history-aware and FEN-only models.

    Preference order:
    - If model declares is_thinking_model, pass move_history
    - Otherwise, pass FEN; on TypeError fall back to history
    """
    if getattr(model, "is_thinking_model", False):
        return model.get_move_from_fen_no_thinking(move_history, T=temp, device=device)
    try:
        return model.get_move_from_fen_no_thinking(board.fen(), T=temp, device=device)
    except TypeError:
        return model.get_move_from_fen_no_thinking(move_history, T=temp, device=device)


def play_game(
    white_name: str,
    black_name: str,
    white_model,
    black_model,
    device: str,
    temp_white: float = 0.1,
    temp_black: float = 0.1,
    game_num: int = 1,
    opening: Optional[Dict[str, Any]] = None,
):
    board = bulletchess.Board()
    move_history: List[str] = []

    # Apply opening if provided
    if opening is not None:
        fen = opening.get("fen")
        if isinstance(fen, str) and fen:
            try:
                board = bulletchess.Board.from_fen(fen)
            except Exception:
                board = bulletchess.Board()
        uci_moves = opening.get("uci") or []
        if isinstance(uci_moves, list):
            move_history = [str(mv) for mv in uci_moves]

    # Setup PGN
    game = pgn.Game()
    game.headers["Event"] = f"{white_name} vs {black_name}"
    game.headers["Site"] = "Local Machine"
    game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
    game.headers["Round"] = str(game_num)
    game.headers["White"] = white_name
    game.headers["Black"] = black_name

    # If starting from an opening FEN, record it for correct PGN export and save opening moves
    if opening is not None:
        fen = opening.get("fen") if isinstance(opening, dict) else None
        if isinstance(fen, str) and fen:
            game.headers["SetUp"] = "1"
            game.headers["FEN"] = fen
        uci_moves = opening.get("uci") if isinstance(opening, dict) else None
        if isinstance(uci_moves, list) and len(uci_moves) > 0:
            game.headers["OpeningUCI"] = " ".join(uci_moves)

    node = game

    while not _is_game_over(board):
        model_to_use = white_model if board.turn == bulletchess.WHITE else black_model
        mover_name = white_name if board.turn == bulletchess.WHITE else black_name

        temp_turn = temp_white if board.turn == bulletchess.WHITE else temp_black
        move_uci = _safe_get_move(model_to_use, board, move_history, temp_turn, device)
        if move_uci is None:
            print(f"No move from model {mover_name} in {board.fen()}. Forfeiting.")
            # Forfeit on failure to move
            result = "0-1" if board.turn == bulletchess.WHITE else "1-0"
            game.headers["Result"] = result
            return result, game

        try:
            move = bulletchess.Move.from_uci(move_uci)
            if move not in board.legal_moves():
                move = bulletchess.Move.from_uci(move_uci+"n")
                print(f"Converted {move_uci} to {move.uci()} for knight promotion")
        except ValueError as e:
            # Malformed UCI -> forfeit
            print(f"Malformed UCI: {move_uci} from model {mover_name} in {board.fen()}: {e}")
            result = "0-1" if board.turn == bulletchess.WHITE else "1-0"
            game.headers["Result"] = result
            return result, game

        board.apply(move)
        move_history.append(move.uci())

        # Add to PGN (convert to python-chess move)
        try:
            ch_move = chess.Move.from_uci(move.uci())
            node = node.add_variation(ch_move)
        except Exception:
            # If conversion fails, still continue the game
            pass

    if board in CHECKMATE:
        result = "0-1" if board.turn == bulletchess.WHITE else "1-0"
    elif board in DRAW:
        result = "1/2-1/2"
    game.headers["Result"] = result
    return result, game


def main_model_vs_model(
    model1,
    model2,
    model1_name: str = "Model1",
    model2_name: str = "Model2",
    device: str = "cuda",
    num_games: int = 50,
    temp1: float = 0.1,
    temp2: float = 0.1,
    use_openings: bool = True,
    seed: Optional[int] = None,
):
    # Openings generator (HF Lichess/chess-openings)
    openings_gen = None
    if use_openings:
        try:
            from src.generate.openings import hf_opening_fen_generator as hf_fen_generator
        except Exception:
            # Optional fallback if project structure differs
            from data.generate.openings import hf_fen_generator  # type: ignore
        openings_gen = hf_fen_generator(split="train", seed=seed)

    pgn_dir = "artifacts/pgns"
    os.makedirs(pgn_dir, exist_ok=True)
    pgn_filename = os.path.join(
        pgn_dir, datetime.now().strftime("brawls_%Y%m%d_%H%M%S.pgn")
    )
    # Create/clear the PGN file for progressive writing
    open(pgn_filename, "w", encoding="utf-8").close()

    wins1 = 0
    wins2 = 0
    draws = 0
    all_games: List[pgn.Game] = []

    for i in range(num_games):
        white = model1 if i % 2 == 0 else model2
        black = model2 if i % 2 == 0 else model1
        white_name = model1_name if i % 2 == 0 else model2_name
        black_name = model2_name if i % 2 == 0 else model1_name

        opening = next(openings_gen) if openings_gen is not None else None

        result, game = play_game(
            white_name,
            black_name,
            white,
            black,
            device,
            temp_white=(temp1 if i % 2 == 0 else temp2),
            temp_black=(temp2 if i % 2 == 0 else temp1),
            game_num=i + 1,
            opening=opening,
        )

        all_games.append(game)

        # Write game to PGN file immediately
        with open(pgn_filename, "a", encoding="utf-8") as f:
            exporter = pgn.FileExporter(f)
            game.accept(exporter)
            f.write("\n\n")

        if result == "1-0":
            if i % 2 == 0:
                wins1 += 1
            else:
                wins2 += 1
        elif result == "0-1":
            if i % 2 == 0:
                wins2 += 1
            else:
                wins1 += 1
        else:
            draws += 1

        played = i + 1
        score1 = wins1 + 0.5 * draws
        score2 = wins2 + 0.5 * draws
        print(
            f"Game {played}/{num_games} done. Score: {model1_name} {score1:.1f} - {score2:.1f} {model2_name}"
        )

    print(
        f"Final: {model1_name} wins={wins1}, {model2_name} wins={wins2}, draws={draws} (saved {len(all_games)} games to {pgn_filename})"
    )


def _load_small_model(device: str, ckpt_path: Optional[str] = None):
    from src.models.small.small_model import SmallModel

    model = SmallModel().to(device)
    if ckpt_path and os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, weights_only=False, map_location=device)
        if isinstance(state, dict) and "model" in state:
            model.load_state_dict(state["model"], strict=False)
        elif isinstance(state, dict) and "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"], strict=False)
        else:
            model.load_state_dict(state, strict=False)
    elif ckpt_path:
        raise FileNotFoundError(f"SmallModel checkpoint not found: {ckpt_path}")
    model.eval()
    return model


def _load_leela_model(device: str, ckpt_path: Optional[str]):
    from src.models.leela import BT4

    model = BT4().to(device)
    # Mark as history-based model so moves use move_history
    setattr(model, "is_thinking_model", True)
    # Default checkpoint for leela model
    default_ckpt = "src/models/leela/model.pt"
    ckpt_to_load = ckpt_path if ckpt_path else default_ckpt
    if ckpt_to_load and os.path.exists(ckpt_to_load):
        state = torch.load(ckpt_to_load)
        if isinstance(state, dict) and "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"]) 
        else:
            model.load_state_dict(state)
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Model vs Model brawls with HF openings")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_games", type=int, default=50)
    # Temperatures: allow global --temp or per-model --temp1/--temp2
    parser.add_argument("--temp", type=float, default=None)
    parser.add_argument("--temp1", type=float, default=None)
    parser.add_argument("--temp2", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no_openings", action="store_true")

    # Model selections: default to small vs leela
    parser.add_argument("--model1", choices=["small", "leela"], default="small")
    parser.add_argument("--model2", choices=["small", "leela"], default="leela")
    parser.add_argument(
        "--ckpt1", default=None, help="Checkpoint path for model1 (defaults by model type)"
    )
    parser.add_argument(
        "--ckpt2", default=None, help="Checkpoint path for model2 (defaults by model type)"
    )
    parser.add_argument("--name1", default=None)
    parser.add_argument("--name2", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = args.device
    # Resolve temperatures with backwards-compatible defaults (default 0.1)
    global_temp = 0.1 if args.temp is None else args.temp
    temp1 = args.temp1 if args.temp1 is not None else global_temp
    temp2 = args.temp2 if args.temp2 is not None else global_temp
    
    if args.model1 == "small":
        model1 = _load_small_model(device, args.ckpt1)
    else:
        model1 = _load_leela_model(device, args.ckpt1)

    if args.model2 == "small":
        model2 = _load_small_model(device, args.ckpt2)
    else:
        model2 = _load_leela_model(device, args.ckpt2)

    # Resolve names from model types if not explicitly provided
    def _default_name(choice: str) -> str:
        return "Small" if choice == "small" else "Leela"

    name1 = args.name1 if args.name1 is not None else _default_name(args.model1)
    name2 = args.name2 if args.name2 is not None else _default_name(args.model2)
    if name1 == name2:
        # Disambiguate identical names
        name1 = f"{name1}1"
        name2 = f"{name2}2"

    main_model_vs_model(
        model1,
        model2,
        model1_name=name1,
        model2_name=name2,
        device=device,
        num_games=args.num_games,
        temp1=temp1,
        temp2=temp2,
        use_openings=not args.no_openings,
        seed=args.seed,
    )


