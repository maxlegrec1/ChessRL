from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import bulletchess
from bulletchess import CHECKMATE, DRAW

from src.mcts import SmallMCTS

from .buffer import SelfPlayBuffer, SelfPlayRecord
from .engine_watch import EngineWatcher
from .metrics import MetricsTracker


SIMULATIONS = 600
CPUCT = 1.0
TEMPERATURE = 1.0
TEMPERATURE_AT_ROOT = 1.4


@dataclass
class BoardSlot:
    board: bulletchess.Board
    ply_count: int = 0


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

        records: List[SelfPlayRecord] = []
        finished_games = 0

        for idx, slot in enumerate(self.boards):
            board = slot.board
            fen = fens[idx]
            outcome = results[idx]

            value = _coerce_value(outcome.get("value"))
            move_uci = _policy_argmax(outcome.get("policy"), outcome.get("action"))

            if move_uci is None:
                legal_moves = list(board.legal_moves())
                if not legal_moves:
                    finished_games += 1
                    self.boards[idx] = BoardSlot(board=bulletchess.Board())
                    continue
                move_uci = legal_moves[0].uci()

            records.append(SelfPlayRecord(fen=fen, value=value, policy=move_uci))

            move = bulletchess.Move.from_uci(move_uci)
            board.apply(move)
            slot.ply_count += 1

            if board in CHECKMATE or board in DRAW:
                finished_games += 1
                self.boards[idx] = BoardSlot(board=bulletchess.Board())

        if records:
            self.buffer.add_records(records)
            self.metrics.record_positions(len(records))
            self.metrics.update_buffer_fill(self.buffer.fill_ratio())

        if finished_games:
            self.metrics.record_game_finished(finished_games)


def _coerce_value(value_obj) -> float:
    if value_obj is None:
        return 0.0
    if isinstance(value_obj, (list, tuple)):
        if not value_obj:
            return 0.0
        return float(value_obj[0])
    return float(value_obj)


def _policy_argmax(policy_dict, fallback_action) -> str | None:
    if isinstance(policy_dict, dict) and policy_dict:
        try:
            return list(policy_dict.keys())[0]
        except (KeyError, IndexError):
            return None
    if isinstance(policy_dict, (list, tuple)) and policy_dict:
        # handle list of (move, prob)
        move = policy_dict[0][0]
        return str(move)
    if fallback_action:
        return str(fallback_action)
    return None


__all__ = ["SelfPlayLoop"]

