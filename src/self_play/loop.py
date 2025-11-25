from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List
from uuid import uuid4

import bulletchess
from bulletchess import CHECKMATE, DRAW
import chess
import chess.pgn

from src.mcts import SmallMCTS

from .buffer import SelfPlayBuffer, SelfPlayRecord
from .engine_watch import EngineWatcher
from .metrics import MetricsTracker


SIMULATIONS = 600
CPUCT = 1.0
TEMPERATURE = 1.0
TEMPERATURE_AT_ROOT = 1.4
USE_DIRICHLET = True
DIRICHLET_ALPHA = 0.3
DIRICHLET_EPSILON = 0.25
MAX_PLY = 600


@dataclass
class BoardSlot:
    board: bulletchess.Board
    ply_count: int = 0
    move_history: List[str] = field(default_factory=list)
    pending_records: List[SelfPlayRecord] = field(default_factory=list)


class SelfPlayLoop:
    """Runs N bulletchess boards through batched SmallMCTS self-play."""

    def __init__(
        self,
        *,
        board_count: int,
        buffer: SelfPlayBuffer,
        metrics: MetricsTracker,
        engine_watcher: EngineWatcher,
    ) -> None:
        self.board_count = board_count
        self.buffer = buffer
        self.metrics = metrics
        self.engine_watcher = engine_watcher
        self.mcts = SmallMCTS(engine_path=engine_watcher.engine_path or "model.trt")
        self._pgn_dir = Path("pgns")
        self._pgn_dir.mkdir(parents=True, exist_ok=True)

        self.boards: List[BoardSlot] = [BoardSlot(board=bulletchess.Board()) for _ in range(board_count)]
        self._logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------ #
    def step(self) -> None:
        """Run a single batch of self-play moves."""
        engine_override = self.engine_watcher.engine_path
        maybe_new = self.engine_watcher.maybe_refresh()
        if maybe_new:
            engine_override = maybe_new
            self.metrics.update_engine_path(maybe_new)

        fens = [slot.board.fen() for slot in self.boards]
        try:
            results = self.mcts.run(
                fens,
                temperature=TEMPERATURE,
                simulations=SIMULATIONS,
                cpuct=CPUCT,
                temperature_at_root=TEMPERATURE_AT_ROOT,
                engine_path=engine_override,
                use_dirichlet=USE_DIRICHLET,
                dirichlet_alpha=DIRICHLET_ALPHA,
                dirichlet_epsilon=DIRICHLET_EPSILON,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            self._logger.exception("SmallMCTS batch failed: %s", exc)
            return

        if not results or len(results) != len(self.boards):
            self._logger.error(
                "Unexpected MCTS result size. expected=%s got=%s",
                len(self.boards),
                0 if not results else len(results),
            )
            return

        records_to_flush: List[SelfPlayRecord] = []
        finished_games = 0

        for idx, slot in enumerate(self.boards):
            board = slot.board
            fen = fens[idx]
            outcome = results[idx]

            # We still get value/policy from MCTS for monitoring/debug, 
            # but we will overwrite value with the game outcome later.
            value = _coerce_value(outcome.get("value"))
            policy_move = _policy_argmax(outcome.get("policy"))
            action = outcome.get("action")
            move_uci = str(action) if action else None

            if move_uci is None:
                self._logger.warning(f"No move found for FEN: {fen}")
                legal_moves = list(board.legal_moves())
                if not legal_moves:
                    # Game over (stalemate or checkmate not caught?)
                    # Treat as finished.
                    self._finish_game(slot, records_to_flush)
                    finished_games += 1
                    continue
                move_uci = legal_moves[0].uci()

            record_policy = policy_move or move_uci

            # knight promotions are considered default promotions
            if record_policy[-1] == "n":
                record_policy = record_policy[:-1]
            else:
                record_policy = record_policy
            
            # Store record in pending list
            slot.pending_records.append(SelfPlayRecord(fen=fen, value=value, policy=record_policy))

            move = bulletchess.Move.from_uci(move_uci)
            board.apply(move)
            slot.ply_count += 1
            slot.move_history.append(move_uci)

            # Check termination
            is_over = (board in CHECKMATE) or (board in DRAW) or (slot.ply_count >= MAX_PLY)
            
            if is_over:
                finished_games += 1
                self._finish_game(slot, records_to_flush)

        if records_to_flush:
            self.buffer.add_records(records_to_flush)
            self.metrics.record_positions(len(records_to_flush))
            self.metrics.update_buffer_fill(self.buffer.fill_ratio())

        if finished_games:
            self.metrics.record_game_finished(finished_games)

    def _finish_game(self, slot: BoardSlot, flush_list: List[SelfPlayRecord]) -> None:
        """Process a finished game: determine outcome, backfill values, and reset slot."""
        board = slot.board
        
        # Determine outcome z from perspective of White
        # 1.0 = White Win, 0.0 = Black Win, 0.5 = Draw
        if board in CHECKMATE:
            # If it's checkmate and it's White's turn, White lost (0.0). 
            # If Black's turn, Black lost (White won, 1.0).
            # Wait, bulletchess.CHECKMATE means the side to move is in checkmate?
            # Usually yes.
            if board.turn == bulletchess.WHITE:
                z = 0.0 # White lost
            else:
                z = 1.0 # White won
        elif (board in DRAW) or (slot.ply_count >= MAX_PLY):
            z = 0.5
        else:
            # Should not happen unless forced termination
            z = 0.5

        self._write_pgn(slot, board)

        # Backfill values
        # Value in SelfPlayRecord is [Win, Draw, Loss] relative to side to move.
        # If side to move is White:
        #   Win=1 -> z=1.0
        #   Loss=1 -> z=0.0
        #   Draw=1 -> z=0.5
        # If side to move is Black:
        #   Win=1 -> z=0.0 (Black wins)
        #   Loss=1 -> z=1.0 (White wins)
        #   Draw=1 -> z=0.5
        
        for record in slot.pending_records:
            # Parse FEN to check turn
            # We can do a quick string check
            is_white_turn = " w " in record.fen
            
            if is_white_turn:
                if z == 1.0:
                    new_val = (1.0, 0.0, 0.0) # Win
                elif z == 0.0:
                    new_val = (0.0, 0.0, 1.0) # Loss
                else:
                    new_val = (0.0, 1.0, 0.0) # Draw
            else:
                if z == 0.0:
                    new_val = (1.0, 0.0, 0.0) # Win (Black wins)
                elif z == 1.0:
                    new_val = (0.0, 0.0, 1.0) # Loss (White wins)
                else:
                    new_val = (0.0, 1.0, 0.0) # Draw
            
            # Create new record since SelfPlayRecord is frozen
            new_record = SelfPlayRecord(
                fen=record.fen,
                value=new_val,
                policy=record.policy
            )
            flush_list.append(new_record)
            
        # Reset slot
        slot.board = bulletchess.Board()
        slot.ply_count = 0
        slot.move_history.clear()
        slot.pending_records.clear()


    # ------------------------------------------------------------------ #
    # PGN helpers
    # ------------------------------------------------------------------ #
    def _write_pgn(self, slot: BoardSlot, final_board: bulletchess.Board) -> None:
        if not slot.move_history:
            return

        game = chess.pgn.Game()
        now = datetime.utcnow()
        game.headers["Event"] = "Self-Play"
        game.headers["Site"] = "Local"
        game.headers["Date"] = now.strftime("%Y.%m.%d")
        game.headers["Round"] = "-"
        game.headers["White"] = "SelfPlay"
        game.headers["Black"] = "SelfPlay"
        game.headers["Result"] = self._result_from_board(final_board)

        node = game
        chess_board = chess.Board()
        try:
            for move_uci in slot.move_history:
                chess_move = chess.Move.from_uci(move_uci)
                chess_board.push(chess_move)
                node = node.add_variation(chess_move)
        except ValueError:
            self._logger.warning("Failed to encode PGN move %s; skipping PGN export", move_uci)
            return

        game_id = uuid4().hex
        pgn_path = self._pgn_dir / f"{game_id}.pgn"
        try:
            with pgn_path.open("w", encoding="utf-8") as handle:
                exporter = chess.pgn.FileExporter(handle)
                game.accept(exporter)
        except OSError as exc:
            self._logger.warning("Failed to write PGN %s: %s", pgn_path, exc)

    def _result_from_board(self, board: bulletchess.Board) -> str:
        if board in CHECKMATE:
            return "0-1" if board.turn == bulletchess.WHITE else "1-0"
        if board in DRAW:
            return "1/2-1/2"
        return "*"


def _coerce_value(value_obj) -> tuple[float, float, float]:

    vals = list(value_obj)
    return (float(vals[0]), float(vals[1]), float(vals[2]))



def _policy_argmax(policy_dict) -> str | None:

    return list(policy_dict.keys())[0]



__all__ = ["SelfPlayLoop"]

