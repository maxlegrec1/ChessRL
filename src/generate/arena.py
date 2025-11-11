from dataclasses import dataclass
from typing import Callable, Iterator, List, Optional, Tuple, Dict, Any
import math
import os
import time
import random
import torch
import bulletchess
import chess

# Types
FenGenerator = Callable[[], Iterator[Any]]


@dataclass
class SelfPlayConfig:
    batch_size: int = 1024
    max_moves: int = 512
    player_1_temperature: float = 0.0
    player_2_temperature: float = 0.0
    device_player_1: str = "cuda"
    device_player_2: str = "cuda"
    player_1_fp16: bool = False
    player_2_fp16: bool = False
    target_games: Optional[int] = None
    target_hours: Optional[float] = None
    output_dir: str = "data/selfplay_pgn"
    swap_sides_every_n: int = 0  # 0 disables
    player_1_name: str = "player-1"
    player_2_name: str = "player-2"
    randomize_starting_player: bool = False
    fen_generator: Optional[FenGenerator] = None
    calibrate: bool = False
    calibration_games: int = 0
    # Legacy controller params (used by second_order mode only)
    calibration_window_size: int = 256
    calibration_lr: float = 0.02
    calibration_deadband: float = 0.05
    calibration_max_step: float = 0.1
    calibration_gain: float = 5.0
    # Bucketed calibration strategy params
    calibration_mode: str = "buckets"  # "buckets" | "second_order"
    calibration_buckets: int = 5
    calibration_samples_per_bucket: int = 256
    calibration_temp_min: float = 0.05
    calibration_temp_max: float = 2.0


class SelfPlayArena:
    def __init__(self, player1_model, player2_model, cfg: SelfPlayConfig):
        self.player1_model = player1_model
        self.player2_model = player2_model
        self.cfg = cfg

        os.makedirs(self.cfg.output_dir, exist_ok=True)

        # No logging for hot path

        self.boards: List[bulletchess.Board] = []
        self.histories: List[List[str]] = []  # UCI move lists per game (during self-play)
        self.opening_histories: List[List[str]] = []  # Opening UCI moves that produce start_fen
        self.start_fens: List[str] = []
        self.done: List[bool] = []
        self.results: List[str] = []  # "*" for unknown, or PGN result string
        self.white_is_p1: List[bool] = []
        self.p2_temp_per_game: List[float] = []

        self.total_moves: int = 0
        self.finished_games: int = 0
        self.started_games: int = 0
        self.start_time_s: float = time.time()

        # Result tallies
        self.player1_wins: int = 0
        self.player2_wins: int = 0
        self.draws: int = 0

        # No profiling state

        # Internal generator state
        self._fen_iter: Optional[Iterator[Any]] = None
        if self.cfg.fen_generator is not None:
            self._fen_iter = self.cfg.fen_generator()

        # Initialize Leela temp with a safe floor and clamp
        _t0 = float(self.cfg.player_2_temperature)
        if _t0 <= 0.0:
            _t0 = 0.7
        self.current_player_2_temp: float = max(0.05, min(2.0, _t0))
        # Calibration orchestrator
        self.calibrating: bool = bool(self.cfg.calibrate and self.cfg.calibration_games > 0)
        self.calib_target: int = int(self.cfg.calibration_games)
        self.calib_finished: int = 0
        self.calib_p2_wins: int = 0
        # History for UI trend plotting (kept short)
        self._wr_hist: List[float] = []
        self._temp_hist: List[float] = [self.current_player_2_temp]
        # Strategy
        self._calibrator = None
        if self.calibrating:
            mode = getattr(self.cfg, "calibration_mode", "buckets")
            if mode == "second_order":
                self._calibrator = _SecondOrderCalibrator(self)
            else:
                self._calibrator = _BucketsCalibrator(self)

    # --------- Public API ---------
    def run(self):
        """Run the arena until targets are met; yields periodic snapshots for UI."""
        refresh_each_steps = 10
        steps = 0
        while not self._targets_reached():
            self._ensure_capacity()
            moved = self.step()
            steps += 1
            if steps % refresh_each_steps == 0:
                yield self.snapshot()
            # If nothing moved and no active games, break to avoid tight loop
            if moved == 0 and self._active_games_count() == 0:
                break
        # Final snapshot
        yield self.snapshot()

    def step(self) -> int:
        """Advance all active games by one ply per side (white then black). Returns number of plies played."""
        moved = 0

        # 1) White to move
        white_idxs = [i for i, b in enumerate(self.boards)
                      if not self.done[i] and b.turn == bulletchess.WHITE]
        # split white indices by which player is white
        white_p1 = [i for i in white_idxs if self.white_is_p1[i]]
        white_p2 = [i for i in white_idxs if not self.white_is_p1[i]]
        moved += self._infer_and_apply(
            white_p1,
            self.player1_model,
            self.cfg.player_1_temperature,
            self.cfg.device_player_1,
            self.cfg.player_1_fp16,
        )
        moved += self._infer_and_apply(
            white_p2,
            self.player2_model,
            self.cfg.player_2_temperature,
            self.cfg.device_player_2,
            self.cfg.player_2_fp16,
        )

        # 2) Black to move
        black_idxs = [i for i, b in enumerate(self.boards)
                      if not self.done[i] and b.turn == bulletchess.BLACK]
        # player 1 is black when white_is_p1 is False
        black_p1 = [i for i in black_idxs if not self.white_is_p1[i]]
        black_p2 = [i for i in black_idxs if self.white_is_p1[i]]
        moved += self._infer_and_apply(
            black_p1,
            self.player1_model,
            self.cfg.player_1_temperature,
            self.cfg.device_player_1,
            self.cfg.player_1_fp16,
        )
        moved += self._infer_and_apply(
            black_p2,
            self.player2_model,
            self.cfg.player_2_temperature,
            self.cfg.device_player_2,
            self.cfg.player_2_fp16,
        )

        return moved

    def snapshot(self) -> Dict[str, float]:
        """Return metrics for UI."""
        elapsed = max(time.time() - self.start_time_s, 1e-6)
        games_per_hour = (self.finished_games / elapsed) * 3600.0
        moves_per_hour = (self.total_moves / elapsed) * 3600.0
        # Compute small winrate using configured names
        p1_winrate = (self.player1_wins + 0.5 * self.draws) / self.finished_games if self.finished_games > 0 else 0.0
        p2_overall = (self.player2_wins + 0.5 * self.draws) / self.finished_games if self.finished_games > 0 else 0.0
        is_calib = bool(getattr(self, "calibrating", False))
        p2_calib = (self.calib_p2_wins / self.calib_finished) if is_calib and self.calib_finished > 0 else 0.0
        # Pre-render simple sparklines for UI
        def _spark(values: List[float], width: int = 40) -> str:
            if not values:
                return ""
            if len(values) > width:
                step = len(values) / width
                xs = [values[int(i * step)] for i in range(width)]
            else:
                xs = list(values)
            vmin, vmax = min(xs), max(xs)
            rng = max(1e-6, vmax - vmin)
            blocks = "▁▂▃▄▅▆▇█"
            out = []
            for v in xs:
                idx = int((v - vmin) / rng * (len(blocks) - 1))
                out.append(blocks[idx])
            return "".join(out)
        wr_trend = _spark(getattr(self, "_wr_hist", []))
        # Normalize temperature history for trend
        th = getattr(self, "_temp_hist", [])
        temp_trend = ""
        if th:
            tmin, tmax = min(th), max(th)
            if tmax - tmin < 1e-6:
                temp_trend = _spark([0.5 for _ in th])
            else:
                temp_trend = _spark([(t - tmin) / (tmax - tmin) for t in th])
        # Pull estimates from calibrator if present
        est_wr = 0.0
        est_win_window = 0.0
        est_temp = None
        buckets_summary = None
        mode = getattr(self.cfg, "calibration_mode", "buckets")
        if getattr(self, "_calibrator", None) is not None:
            wrv = self._calibrator.estimated_wr()
            if wrv is not None:
                est_wr = float(wrv)
            winw = getattr(self._calibrator, "effective_window", None)
            if callable(winw):
                ev = self._calibrator.effective_window()
                if ev is not None:
                    est_win_window = float(ev)
            et = self._calibrator.estimated_temp()
            if et is not None:
                est_temp = float(et)
            # Buckets exposure
            if mode == "buckets" and isinstance(self._calibrator, _BucketsCalibrator):
                temps = list(self._calibrator.bucket_temps)
                games = list(self._calibrator.bucket_games)
                wrs: List[float] = []
                for s, g in zip(self._calibrator.bucket_score_sum, self._calibrator.bucket_games):
                    wrs.append((float(s) / float(g)) if g > 0 else 0.0)
                brackets = getattr(self._calibrator, "_bracket", None)
                buckets_summary = {
                    "temps": [float(t) for t in temps],
                    "games": [int(x) for x in games],
                    "wrs": wrs,
                    "samples_per_bucket": int(self._calibrator.samples_per_bucket),
                    "bracket_lo": float(brackets[0]) if brackets else None,
                    "bracket_hi": float(brackets[1]) if brackets else None,
                }
        # Reported temperature: during calibration show current estimate; after, show calibrated temp
        # Use current_player_2_temp which is always updated to the calibrated value when calibration finishes
        reported_temp = self.current_player_2_temp
        return {
            "elapsed_s": elapsed,
            "elapsed_h": elapsed / 3600.0,
            "finished_games": float(self.finished_games),
            "started_games": float(self.started_games),
            "active_games": float(self._active_games_count()),
            "total_moves": float(self.total_moves),
            "games_per_hour": games_per_hour,
            "moves_per_hour": moves_per_hour,
            # Names
            "player_1_name": self.cfg.player_1_name,
            "player_2_name": self.cfg.player_2_name,
            # Results
            "player_1_wins": float(self.player1_wins),
            "player_2_wins": float(self.player2_wins),
            "draws": float(self.draws),
            "player_1_winrate": p1_winrate,
            # Calibration/Leela temp
            "calibrating": float(1.0 if getattr(self, "calibrating", False) else 0.0),
            "calibration_done": float(getattr(self, "calib_finished", 0)),
            "calibration_target": float(getattr(self, "calib_target", 0)),
            "player_2_temp": float(reported_temp),
            "estimated_temp": float(est_temp) if est_temp is not None else float(getattr(self, "current_player_2_temp", self.cfg.player_2_temperature)),
            "player_2_winrate_calib": p2_calib,
            "player_2_winrate_overall": p2_overall,
            "player_2_running_wr": est_wr,
            "player_2_running_wr_window": est_win_window,
            "wr_trend": wr_trend,
            "temp_trend": temp_trend,
            "calibration_mode": mode,
            "calibration_buckets": buckets_summary if buckets_summary is not None else None,
        }

    # --------- Internal helpers ---------
    def _ensure_capacity(self):
        # Maintain at least batch_size active games, spawn as needed
        need = max(0, self.cfg.batch_size - self._active_games_count())
        if need > 0:
            self._spawn_games(need)

    def _spawn_games(self, n: int):
        for _ in range(n):
            fen, hist = self._next_opening()
            board = bulletchess.Board.from_fen(fen)
            self.boards.append(board)
            # Store opening history separately; in-game history starts empty
            self.opening_histories.append(list(hist))
            self.histories.append([])
            self.start_fens.append(fen)
            self.done.append(False)
            self.results.append("*")
            self.started_games += 1
            if self.cfg.randomize_starting_player:
                self.white_is_p1.append(bool(random.getrandbits(1)))
            else:
                self.white_is_p1.append(True)
            # Record per-game temp for Player 2 (Leela)
            if self.calibrating and self._calibrator is not None:
                t_new = self._calibrator.assign_temp_for_new_game(len(self.boards) - 1)
                self.p2_temp_per_game.append(t_new)
            else:
                self.p2_temp_per_game.append(self.cfg.player_2_temperature)

    def _next_opening(self) -> Tuple[str, List[str]]:
        if self._fen_iter is not None:
            try:
                item = next(self._fen_iter)
                if isinstance(item, str):
                    return item, []
                if isinstance(item, dict):
                    fen = str(item.get("fen", bulletchess.Board().fen()))
                    uci = item.get("uci")
                    if not isinstance(uci, list):
                        uci = []
                    # validate move list length parity (we want Black to have moved as well, but not required)
                    return fen, list(uci)
                return str(item), []
            except StopIteration:
                self._fen_iter = None
        return bulletchess.Board().fen(), []

    def _infer_and_apply(self, indices: List[int], model, T: float, device: str, use_fp16: bool) -> int:
        if not indices:
            return 0

        plies = 0
        # Process in chunks to respect batch_size and reduce peak mem
        for chunk_start in range(0, len(indices), self.cfg.batch_size):
            chunk = indices[chunk_start:chunk_start + self.cfg.batch_size]
            # Filter out boards that already ended (in case of overlap)
            chunk = [i for i in chunk if not self.done[i]]
            if not chunk:
                continue

            # Build FEN list; skip boards with no legal moves (mark done)
            fens: List[str] = []
            valid_idx: List[int] = []
            for i in chunk:
                if not _has_legal_moves(self.boards[i]):
                    self._finalize_game(i)
                    continue
                fens.append(self.boards[i].fen())
                valid_idx.append(i)

            if not valid_idx:
                continue

            # Batched inference
            moves: List[str] = [""] * len(valid_idx)
            is_leela = _is_leela_model(model)
            use_moves_api = is_leela and all((len(self.opening_histories[i]) + len(self.histories[i])) > 0 for i in valid_idx)
            if is_leela:
                # Group by per-game fixed temp for Leela
                temp_to_indices: Dict[float, List[int]] = {}
                for gi in valid_idx:
                    # Use per-game temp if available (set during calibration), otherwise use config temp
                    if gi < len(self.p2_temp_per_game):
                        tgi = self.p2_temp_per_game[gi]
                    else:
                        tgi = self.cfg.player_2_temperature
                    temp_to_indices.setdefault(float(tgi), []).append(gi)
                for tval, idxs in temp_to_indices.items():
                    if use_moves_api:
                        move_lists = [self.opening_histories[i] + self.histories[i] for i in idxs]
                        gm = model.batch_get_moves_from_move_lists(move_lists, T=tval, device=device, use_fp16=use_fp16)
                    else:
                        fgrp = [self.boards[i].fen() for i in idxs]
                        gm = model.batch_get_moves_from_fens(fgrp, T=tval, device=device, use_fp16=use_fp16)
                    for gi, mi in zip(idxs, gm):
                        pos = valid_idx.index(gi)
                        moves[pos] = mi
            else:
                if use_moves_api:
                    move_lists = [self.opening_histories[i] + self.histories[i] for i in valid_idx]
                    gm = model.batch_get_moves_from_move_lists(move_lists, T=T, device=device, use_fp16=use_fp16)
                else:
                    gm = model.batch_get_moves_from_fens(fens, T=T, device=device, use_fp16=use_fp16)
                for pos, mi in enumerate(gm):
                    moves[pos] = mi

            # Apply moves
            for game_idx, uci in zip(valid_idx, moves):
                board = self.boards[game_idx]
                try:
                    # Auto-promote to knight on missing promotion piece to ensure legality
                    if _looks_like_missing_promotion(uci, board):
                        uci = uci + 'n'
                    mv = bulletchess.Move.from_uci(uci)
                    board.apply(mv)
                    self.histories[game_idx].append(uci)
                    self.total_moves += 1
                    plies += 1
                except Exception as e:
                    # If something goes wrong, end the game gracefully
                    self._finalize_game(game_idx)
                    continue

                # End conditions: no legal moves or max_moves reached
                if (not _has_legal_moves(board)) or (len(self.histories[game_idx]) >= self.cfg.max_moves):
                    self._finalize_game(game_idx)


        return plies

    def _finalize_game(self, i: int):
        if self.done[i]:
            return
        self.done[i] = True
        self.finished_games += 1
        # Write PGN immediately
        try:
            # Compute result from python-chess from startpos using full move list (opening + game)
            res = "*"
            try:
                b = chess.Board()
                for u in (self.opening_histories[i] + self.histories[i]):
                    mv = chess.Move.from_uci(u)
                    if mv in b.legal_moves:
                        b.push(mv)
                    else:
                        break
                outcome = b.outcome()
                if outcome is not None:
                    if outcome.winner is None:
                        res = "1/2-1/2"
                    else:
                        res = "1-0" if outcome.winner else "0-1"
            except Exception:
                res = "*"
            self.results[i] = res
            # Update per-player tallies using mapping for this game
            if res == "1-0":
                if self.white_is_p1[i]:
                    self.player1_wins += 1
                else:
                    self.player2_wins += 1
            elif res == "0-1":
                if self.white_is_p1[i]:
                    self.player2_wins += 1
                else:
                    self.player1_wins += 1
            elif res == "1/2-1/2":
                self.draws += 1

            # Calibration: update stats through calibrator
            if self.cfg.calibrate and self.cfg.calibration_games > 0 and self.calibrating and self._calibrator is not None:
                self.calib_finished += 1
                p2_won = (res == "1-0" and not self.white_is_p1[i]) or (res == "0-1" and self.white_is_p1[i])
                is_draw = (res == "1/2-1/2")
                score = 1.0 if p2_won else (0.5 if is_draw else 0.0)
                if p2_won:
                    self.calib_p2_wins += 1
                self._calibrator.on_game_finished(i, score)
                # Update UI histories
                wr_est = self._calibrator.estimated_wr()
                if wr_est is not None:
                    self._wr_hist.append(wr_est)
                    if len(self._wr_hist) > 120:
                        self._wr_hist.pop(0)
                est_t = self._calibrator.estimated_temp()
                if est_t is not None:
                    self.current_player_2_temp = est_t
                self._temp_hist.append(self.current_player_2_temp)
                if len(self._temp_hist) > 120:
                    self._temp_hist.pop(0)
                # Finish?
                if self.calib_finished >= self.calib_target or self._calibrator.is_done():
                    self.calibrating = False
                    # Always calculate directly from bucket data to avoid issues with stale _est_temp
                    # This is more reliable than using estimated_temp() which may be incorrect
                    final_t = None
                    if hasattr(self._calibrator, 'bucket_temps') and hasattr(self._calibrator, 'bucket_games'):
                        pts = []
                        for t, g, s in zip(self._calibrator.bucket_temps, 
                                          self._calibrator.bucket_games,
                                          self._calibrator.bucket_score_sum):
                            if g > 0:
                                wr = float(s) / float(g)
                                pts.append((float(t), wr))
                        print(f"  DEBUG: Calculated points: {pts}")
                        if len(pts) >= 2:
                            pts.sort(key=lambda x: x[0])  # Sort by temperature
                            print(f"  DEBUG: Sorted points: {pts}")
                            # Try to find crossing of 0.5 win rate
                            est_t_calc = None
                            for i, ((t1, wr1), (t2, wr2)) in enumerate(zip(pts, pts[1:])):
                                # Check if win rate crosses 0.5 between these two buckets
                                product = (wr1 - 0.5) * (wr2 - 0.5)
                                print(f"  DEBUG: Pair {i}: ({t1:.3f}, {wr1:.3f}) and ({t2:.3f}, {wr2:.3f}), product={product:.6f}")
                                if product < 0:
                                    # Linear interpolation to find temp where wr = 0.5
                                    # Handle both increasing and decreasing win rate cases
                                    wr_diff = wr2 - wr1
                                    if abs(wr_diff) < 1e-6:
                                        # Win rates are too close, can't interpolate
                                        est_t_calc = 0.5 * (t1 + t2)
                                        print(f"  DEBUG: Found crossing but wr_diff too small! wr_diff={wr_diff:.6f}, using midpoint: est_t_calc={est_t_calc:.6f}")
                                    else:
                                        alpha = (0.5 - wr1) / wr_diff
                                        est_t_calc = t1 + alpha * (t2 - t1)
                                        print(f"  DEBUG: Found crossing! wr_diff={wr_diff:.6f}, alpha={alpha:.6f}, est_t_calc={est_t_calc:.6f}")
                                    break
                            if est_t_calc is None:
                                # No crossing found, use bucket closest to 0.5
                                best = min(pts, key=lambda x: abs(x[1] - 0.5))
                                est_t_calc = best[0]
                                print(f"  DEBUG: No crossing, using closest bucket: temp={est_t_calc:.6f}, wr={best[1]:.6f}")
                            if est_t_calc is not None:
                                final_t_before_clamp = est_t_calc
                                final_t = float(max(self.cfg.calibration_temp_min, min(self.cfg.calibration_temp_max, est_t_calc)))
                                print(f"  DEBUG: Before clamp: {final_t_before_clamp:.6f}, after clamp: {final_t:.6f}, min={self.cfg.calibration_temp_min}, max={self.cfg.calibration_temp_max}")
                        else:
                            print(f"  DEBUG: Not enough points ({len(pts)}), need at least 2")
                    else:
                        print(f"  DEBUG: Calibrator missing bucket_temps or bucket_games attributes")
                    
                    # Fallback to calibrator methods if direct calculation failed
                    if final_t is None:
                        final_t = self._calibrator.estimated_temp()
                    if final_t is None:
                        final_t = self._calibrator.best_temp()
                    if final_t is None:
                        # Last resort: use current estimate
                        final_t = self.current_player_2_temp
                    
                    # Debug: print calibration results
                    print(f"\n{'='*60}")
                    print(f"CALIBRATION COMPLETE")
                    print(f"  Finished games: {self.calib_finished}/{self.calib_target}")
                    if hasattr(self._calibrator, 'bucket_temps') and hasattr(self._calibrator, 'bucket_games'):
                        print(f"  Bucket results:")
                        for i, (t, g, s) in enumerate(zip(self._calibrator.bucket_temps, 
                                                           self._calibrator.bucket_games,
                                                           self._calibrator.bucket_score_sum)):
                            wr = (s / g) if g > 0 else 0.0
                            print(f"    Bucket {i}: temp={t:.3f}, games={g}, wr={wr:.3f}")
                    est_t_debug = self._calibrator.estimated_temp()
                    best_t_debug = self._calibrator.best_temp()
                    print(f"  estimated_temp() = {est_t_debug}")
                    print(f"  best_temp() = {best_t_debug}")
                    print(f"  Direct calculation from buckets = {final_t}")
                    print(f"  Selected final_t = {final_t}")
                    print(f"{'='*60}\n")
                    
                    # Always update both - this ensures the calibrated temperature is preserved
                    # Only update if we got a valid temperature (not None and > 0)
                    if final_t is not None and final_t > 0:
                        self.current_player_2_temp = float(final_t)
                        self.cfg.player_2_temperature = float(final_t)
                        print(f"  ✓ Temperature set to {final_t:.4f} (stored in both current_player_2_temp and cfg.player_2_temperature)")
                    else:
                        print(f"  ✗ WARNING: Failed to set calibrated temperature! final_t={final_t}")
                else:
                    return

            from .pgn_writer import write_pgn
            game_id = f"game_{int(time.time()*1000)}_{i}"
            out_path = os.path.join(self.cfg.output_dir, f"{game_id}.pgn")
            white_name = self.cfg.player_1_name if self.white_is_p1[i] else self.cfg.player_2_name
            black_name = self.cfg.player_2_name if self.white_is_p1[i] else self.cfg.player_1_name
            write_pgn(
                moves_uci=(self.opening_histories[i] + self.histories[i]),
                out_path=out_path,
                white_name=white_name,
                black_name=black_name,
                result=self.results[i],
                start_fen=chess.STARTING_FEN,
                opening_uci=self.opening_histories[i],
            )
        except Exception:
            # Swallow PGN errors to avoid stopping the arena
            pass

    def _active_games_count(self) -> int:
        return sum(1 for d in self.done if not d)

    def _targets_reached(self) -> bool:
        if self.cfg.target_games is not None and self.finished_games >= self.cfg.target_games:
            return True
        if self.cfg.target_hours is not None:
            elapsed_h = (time.time() - self.start_time_s) / 3600.0
            if elapsed_h >= self.cfg.target_hours:
                return True
        return False


def _has_legal_moves(board: bulletchess.Board) -> bool:
    try:
        for _ in board.legal_moves():
            return True
        return False
    except Exception:
        return False


def _looks_like_missing_promotion(uci: str, board: bulletchess.Board) -> bool:
    # Promotion must include a 5th char (q,r,b,n). If absent on a pawn move reaching last rank, flag.
    if len(uci) != 4:
        return False
    try:
        from_file = ord(uci[0]) - ord('a')
        from_rank = int(uci[1])
        to_file = ord(uci[2]) - ord('a')
        to_rank = int(uci[3])
    except Exception:
        return False
    if not (0 <= from_file <= 7 and 1 <= from_rank <= 8 and 0 <= to_file <= 7 and 1 <= to_rank <= 8):
        return False
    from_idx = (from_rank - 1) * 8 + from_file
    from_sq = bulletchess.SQUARES[from_idx]
    piece = board[from_sq]
    if piece is None or piece.piece_type != bulletchess.PAWN:
        return False
    if board.turn == bulletchess.WHITE and from_rank == 7 and to_rank == 8:
        return True
    if board.turn == bulletchess.BLACK and from_rank == 2 and to_rank == 1:
        return True
    return False


def _is_leela_model(model) -> bool:
    mod = getattr(model.__class__, "__module__", "")
    return "models.leela" in mod


def _is_small_model(model) -> bool:
    mod = getattr(model.__class__, "__module__", "")
    return "models.small" in mod


# ---------------- Calibration strategies ----------------

class _BucketsCalibrator:
    def __init__(self, arena: "SelfPlayArena"):
        self.arena = arena
        cfg = arena.cfg
        self.num_buckets: int = max(2, int(getattr(cfg, "calibration_buckets", 5)))
        self.samples_per_bucket: int = max(16, int(getattr(cfg, "calibration_samples_per_bucket", 256)))
        self.tmin: float = float(getattr(cfg, "calibration_temp_min", 0.05))
        self.tmax: float = float(getattr(cfg, "calibration_temp_max", 2.0))
        # Temps evenly spaced initially
        if self.num_buckets == 1:
            temps = [max(0.05, min(2.0, arena.current_player_2_temp))]
        else:
            temps = [self.tmin + i * (self.tmax - self.tmin) / (self.num_buckets - 1) for i in range(self.num_buckets)]
        self.bucket_temps: List[float] = temps
        self.bucket_score_sum: List[float] = [0.0 for _ in range(self.num_buckets)]
        self.bucket_games: List[int] = [0 for _ in range(self.num_buckets)]
        self.game_to_bucket: Dict[int, int] = {}
        self.next_bucket: int = 0
        self._est_temp: Optional[float] = None
        self._est_wr: Optional[float] = None
        self._bracket: Optional[Tuple[float, float]] = (self.tmin, self.tmax)
        self._recenter_rounds: int = 0
        self._max_recenter_rounds: int = 4

    def assign_temp_for_new_game(self, game_idx: int) -> float:
        b = self.next_bucket % self.num_buckets
        self.next_bucket += 1
        self.game_to_bucket[game_idx] = b
        return float(self.bucket_temps[b])

    def on_game_finished(self, game_idx: int, p2_score: float):
        b = self.game_to_bucket.get(game_idx)
        if b is None:
            return
        self.bucket_score_sum[b] += float(p2_score)
        self.bucket_games[b] += 1
        # Update estimate when a bucket gains a sample
        self._update_estimate_and_maybe_recenter()

    def _update_estimate_and_maybe_recenter(self):
        # Build (temp, wr, games) for buckets with any games
        pts: List[Tuple[float, float, int]] = []
        for t, s, g in zip(self.bucket_temps, self.bucket_score_sum, self.bucket_games):
            if g > 0:
                pts.append((float(t), float(s) / float(g), int(g)))
        if len(pts) < 1:
            return
        pts.sort(key=lambda x: x[0])
        # Try to find crossing of 0.5
        est_t = None
        est_wr = None
        for (t1, wr1, g1), (t2, wr2, g2) in zip(pts, pts[1:]):
            if abs(wr1 - 0.5) < 1e-6:
                est_t, est_wr = t1, wr1
                break
            if abs(wr2 - 0.5) < 1e-6:
                est_t, est_wr = t2, wr2
                break
            # If wr increases with temp, crossing occurs when wr1<0.5<w r2 or vice versa
            # Check if they're on opposite sides of 0.5
            if (wr1 - 0.5) * (wr2 - 0.5) < 0:
                # Linear interpolation
                alpha = (0.5 - wr1) / max(1e-6, (wr2 - wr1))
                est_t = t1 + alpha * (t2 - t1)
                est_wr = 0.5
                # Update bracket
                self._bracket = (min(t1, t2), max(t1, t2))
                break
        if est_t is None:
            # No crossing in sampled temps; pick closest to 0.5
            best = min(pts, key=lambda x: abs(x[1] - 0.5))
            est_t, est_wr = best[0], best[1]
            # Try to form a bracket with neighbor if available
            idx = [p[0] for p in pts].index(est_t)
            if idx > 0:
                self._bracket = (pts[idx-1][0], pts[idx][0])
            elif idx + 1 < len(pts):
                self._bracket = (pts[idx][0], pts[idx+1][0])
            else:
                self._bracket = (self.tmin, self.tmax)
        
        # Ensure est_t is valid before clamping
        if est_t is None:
            # Should never happen if pts has data, but handle gracefully
            return
        
        self._est_temp = float(max(self.tmin, min(self.tmax, est_t)))
        self._est_wr = float(est_wr if est_wr is not None else 0.5)

        # Recenter buckets into narrower bracket once all buckets reached sample target
        if self._recenter_rounds < self._max_recenter_rounds and all(g >= self.samples_per_bucket for g in self.bucket_games):
            lo, hi = self._bracket if self._bracket is not None else (self.tmin, self.tmax)
            if hi - lo > 1e-3:
                if self.num_buckets == 1:
                    new_temps = [self._est_temp]
                else:
                    new_temps = [lo + i * (hi - lo) / (self.num_buckets - 1) for i in range(self.num_buckets)]
                self.bucket_temps = [float(max(self.tmin, min(self.tmax, t))) for t in new_temps]
                # Reset stats for next round of sampling
                self.bucket_score_sum = [0.0 for _ in range(self.num_buckets)]
                self.bucket_games = [0 for _ in range(self.num_buckets)]
                self.next_bucket = 0
                self._recenter_rounds += 1

    def estimated_temp(self) -> Optional[float]:
        return self._est_temp

    def estimated_wr(self) -> Optional[float]:
        return self._est_wr

    def effective_window(self) -> Optional[int]:
        return sum(self.bucket_games) if self.bucket_games else 0

    def best_temp(self) -> Optional[float]:
        if self._est_temp is not None:
            return self._est_temp
        if self._bracket is not None:
            lo, hi = self._bracket
            return 0.5 * (float(lo) + float(hi))
        if self.bucket_temps:
            return float(self.bucket_temps[max(0, min(len(self.bucket_temps)-1, self.next_bucket-1))])
        return None

    def is_done(self) -> bool:
        if self._bracket is None:
            return False
        lo, hi = self._bracket
        return (hi - lo) <= 0.01  # close enough


class _SecondOrderCalibrator:
    def __init__(self, arena: "SelfPlayArena"):
        self.arena = arena
        cfg = arena.cfg
        self.window: int = max(8, int(getattr(cfg, "calibration_window_size", 256)))
        self.deadband: float = float(getattr(cfg, "calibration_deadband", 0.05))
        self.gain: float = float(getattr(cfg, "calibration_gain", 5.0))
        self.max_step: float = float(getattr(cfg, "calibration_max_step", 0.1))
        self.log_t: float = math.log(max(0.05, min(2.0, arena.current_player_2_temp)))
        self.vel: float = 0.0
        self.scores: List[float] = []
        self._est_temp: Optional[float] = math.exp(self.log_t)
        self._est_wr: Optional[float] = None

    def assign_temp_for_new_game(self, game_idx: int) -> float:
        # Use current temp for all new games
        return float(math.exp(self.log_t))

    def on_game_finished(self, game_idx: int, p2_score: float):
        self.scores.append(float(p2_score))
        if len(self.scores) > self.window:
            self.scores.pop(0)
        wr = sum(self.scores) / max(1, len(self.scores))
        self._est_wr = wr
        err = wr - 0.5
        if abs(err) <= self.deadband:
            return
        eff_err = math.copysign(max(0.0, abs(err) - self.deadband), err)
        # Critically damped second-order step in log space
        omega = 0.5 * self.gain  # speed factor
        a = 2.0 * omega
        b = omega * omega
        # Discrete update (dt=1 per finished game)
        self.vel += -a * self.vel - b * eff_err
        dlog = self.vel
        dlog = max(-self.max_step, min(self.max_step, dlog))
        self.log_t = max(math.log(0.05), min(math.log(2.0), self.log_t + dlog))
        self._est_temp = math.exp(self.log_t)

    def estimated_temp(self) -> Optional[float]:
        return self._est_temp

    def estimated_wr(self) -> Optional[float]:
        return self._est_wr

    def effective_window(self) -> Optional[int]:
        return len(self.scores)

    def best_temp(self) -> Optional[float]:
        return self._est_temp

    def is_done(self) -> bool:
        return False
