from __future__ import annotations

import sys
import time
import subprocess
from collections import deque
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from rich.console import Console
from torch import amp, optim

import wandb
import bulletchess

from src.models.small.small_model import SmallModel
from src.models.small.vocab import policy_to_idx

from .config import TrainingConfig
from .dataset import NULL_POLICY_INDEX, RECORD_SIZE, SelfPlayBatch, SelfPlayDataset


AMP_DTYPES: Dict[str, torch.dtype] = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


class Trainer:
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.console = Console()
        self.device = torch.device(config.device)
        self.device_type = config.device
        self.dataset = SelfPlayDataset(
            config.data.dataset_path,
            reload_seconds=config.data.reload_seconds,
        )

        self.model = SmallModel().to(self.device)
        self.model.train()

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2),
        )
        self.scaler = amp.GradScaler(device = self.device)

        self.amp_dtype = AMP_DTYPES.get(config.amp_dtype.lower(), torch.float16)
        self.checkpoint_dir = config.checkpoint.dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.step = 0
        self.samples_processed = 0
        self.start_time = time.time()
        self.last_log_time = self.start_time
        self._current_stockfish_elo = 1400

        self._wandb_run = self._init_wandb()

        # Initialize teacher model
        self.teacher_model = SmallModel().to(self.device)
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        # Load or create initial teacher
        initial_teacher_path = self.checkpoint_dir / "initial_teacher.pt"
        if initial_teacher_path.exists():
            self.console.log(f"Loading initial teacher from {initial_teacher_path}")
            ckpt = torch.load(initial_teacher_path, map_location=self.device,weights_only=False)
            self.teacher_model.load_state_dict(ckpt["model"])
        else:
            self.console.log(f"Creating initial teacher at {initial_teacher_path}")
            # Initialize teacher with current model weights
            self.teacher_model.load_state_dict(self.model.state_dict())
            ckpt = {
                "model": self.teacher_model.state_dict(),
                "config": self.config.to_dict(),
                "step": 0,
            }
            torch.save(ckpt, initial_teacher_path)

        if config.resume_path:
            self._load_checkpoint(config.resume_path)

    # ------------------------------------------------------------------ #
    def _compute_legal_move_mask(self, fen: str, device: torch.device) -> torch.Tensor:
        """Create a binary mask (1 for legal moves, 0 for illegal) for a given FEN."""
        board = bulletchess.Board.from_fen(fen)
        mask = torch.zeros(1929, dtype=torch.float32, device=device)
        
        for move in board.legal_moves():
            uci = move.uci()
            # Handle knight promotions: strip 'n' suffix
            if uci.endswith('n'):
                uci = uci[:-1]
            
            if uci in policy_to_idx:
                idx = policy_to_idx[uci]
                mask[idx] = 1.0
        
        return mask

    # ------------------------------------------------------------------ #
    def _compute_legal_move_probability(
        self, policy_probs: torch.Tensor, fens: list[str]
    ) -> float:
        """
        Compute the probability that the network plays a legal move.
        
        For each position:
        1. Compute softmax of logits (already done, passed as policy_probs)
        2. Create legal moves mask
        3. Multiply softmax with mask
        4. Sum to get probability mass on legal moves
        
        Returns the average across the batch.
        """
        # only compute on the first 10 positions
        fens = fens[:10]
        policy_probs = policy_probs[:10]
        B = policy_probs.shape[0]
        legal_probs = []
        
        for i in range(B):
            legal_mask = self._compute_legal_move_mask(fens[i], policy_probs.device)
            # Check if there are any legal moves (should always be true, but safety check)
            if legal_mask.sum() > 0:
                # Multiply policy probabilities with legal mask and sum
                prob_on_legal = (policy_probs[i] * legal_mask).sum().item()
                legal_probs.append(prob_on_legal)
            else:
                # Position with no legal moves (shouldn't happen in training, but handle gracefully)
                legal_probs.append(0.0)
        
        return sum(legal_probs) / len(legal_probs) if legal_probs else 0.0

    # ------------------------------------------------------------------ #
    def _init_wandb(self):
        if not self.config.wandb.enabled:
            return None
        wandb_dir = self.config.wandb.dir
        if wandb_dir is not None:
            Path(wandb_dir).mkdir(parents=True, exist_ok=True)
        run = wandb.init(
            project=self.config.wandb.project,
            entity=self.config.wandb.entity,
            name=self.config.wandb.run_name or self.config.run_name,
            config=self.config.to_dict(),
            dir=str(wandb_dir) if wandb_dir is not None else None,
        )
        return run

    # ------------------------------------------------------------------ #
    def _wait_for_warmup(self) -> None:
        threshold = self.config.data.min_samples_to_start
        if threshold <= 0:
            return
        poll = max(1.0, float(self.config.data.warmup_poll_seconds))
        buffer_path = self.config.data.dataset_path

        def _live_size() -> int:
            try:
                return buffer_path.stat().st_size // RECORD_SIZE
            except FileNotFoundError:
                return 0

        if _live_size() >= threshold:
            self.dataset.reload(force=True)
            return
        self.console.log(
            f"[warmup] waiting for buffer to reach {threshold:,} samples "
            f"(currently {_live_size():,})"
        )
        # Cheap file-size probe — avoids re-parsing the entire 5 GB buffer
        # every poll. We only do a real reload once the threshold is hit.
        while _live_size() < threshold:
            time.sleep(poll)
            self.console.log(
                f"[warmup] buffer at {_live_size():,}/{threshold:,}"
            )
        self.dataset.reload(force=True)
        self.console.log("[warmup] threshold reached; starting training")

    def run(self) -> None:
        self._wait_for_warmup()
        self.console.log(
            f"Starting training on {self.device} with dataset size {self.dataset.size}"
        )

        target_k = max(0.1, float(self.config.data.target_k))
        buffer_path = self.config.data.dataset_path
        writes_sidecar = buffer_path.parent / (buffer_path.name + ".writes")

        def _live_record_count() -> int:
            """Buffer file size / record size. Used for the warmup gate."""
            try:
                return buffer_path.stat().st_size // RECORD_SIZE
            except FileNotFoundError:
                return self.dataset.size

        def _live_total_writes() -> int:
            """Monotonic write counter from the C++ sidecar.

            Counts every write self-play emits — appends *and* random
            replacements — so it keeps growing after the buffer hits its cap
            (when file size goes flat). Falls back to file-size-based count
            for old runs without the sidecar.
            """
            try:
                with writes_sidecar.open("r") as fh:
                    return int(fh.read().strip())
            except (FileNotFoundError, ValueError):
                return _live_record_count()

        # Sliding-window measurement of g. Records are flushed in BURSTS at
        # game ends, so per-step instant rates spike or read 0 depending on
        # whether the step window caught a flush. The 30s window absorbs
        # bursts the same way `iostat` does. We track (time, total_writes)
        # snapshots — total_writes is monotonic and keeps moving even after
        # the buffer is capped.
        g_window_seconds = 30.0
        g_history: deque = deque()  # (time, total_writes)
        _t0 = time.time()
        g_history.append((_t0, _live_total_writes()))

        try:
            while True:
                self.dataset.reload()
                t0 = time.time()
                batch = self.dataset.sample_batch(self.config.batch_size)
                metrics = self._train_step(batch)
                t1 = time.time()

                # Update sliding window using the monotonic write counter.
                cur_writes = _live_total_writes()
                g_history.append((t1, cur_writes))
                # Cap deque memory at ~5 minutes of snapshots; never pop down
                # to a single entry (which collapses window_dt to ~0).
                cutoff_age = max(g_window_seconds * 4, 300.0)
                while len(g_history) > 2 and (t1 - g_history[0][0]) > cutoff_age:
                    g_history.popleft()
                # Anchor selection: walk from oldest, keep the freshest
                # snapshot that is still >= g_window_seconds old. Falls back
                # to the oldest available entry if all are younger.
                anchor_t, anchor_w = g_history[0]
                for ts, w in g_history:
                    if t1 - ts < g_window_seconds:
                        break
                    anchor_t, anchor_w = ts, w
                window_dt = max(1e-3, t1 - anchor_t)
                g = max(0.0, (cur_writes - anchor_w) / window_dt)

                # Pace: sleep so that s = batch_size / step_interval = target_k * g.
                # Until the window is fully populated (first ~30s of training)
                # we conservatively cap the inferred g at a small value so the
                # trainer doesn't burst on a single 1s burst-flush reading.
                if window_dt < g_window_seconds * 0.5:
                    g_for_pacing = min(g, 50.0)
                else:
                    g_for_pacing = g
                target_step_interval = self.config.batch_size / max(1e-3, target_k * max(g_for_pacing, 1.0))
                step_dt = t1 - t0
                sleep_for = max(0.0, target_step_interval - step_dt)
                time.sleep(sleep_for)

                self.step += 1
                self.samples_processed += self.config.batch_size

                # Effective sample rate after pacing.
                effective_step_interval = max(step_dt, target_step_interval)
                s_eff = self.config.batch_size / max(1e-6, effective_step_interval)
                k_actual = s_eff / max(1e-6, g)
                metrics["pacing/g_records_per_sec"] = float(g)
                metrics["pacing/s_records_per_sec"] = float(s_eff)
                metrics["pacing/k_actual"] = float(k_actual)
                metrics["pacing/sleep_s"] = float(sleep_for)
                metrics["pacing/step_time_s"] = float(step_dt)
                metrics["pacing/g_window_s"] = float(window_dt)

                if self.step % self.config.log_every == 0:
                    self._log(metrics)

                if self.step % self.config.checkpoint.every_steps == 0:
                    self._save_checkpoint()
        except KeyboardInterrupt:
            self.console.log("Training interrupted by user.")
        finally:
            if self._wandb_run:
                wandb.finish()

    # ------------------------------------------------------------------ #
    def _train_step(self, batch: SelfPlayBatch) -> Dict[str, float]:
        self.optimizer.zero_grad(set_to_none=True)

        inputs = torch.from_numpy(batch.planes).to(self.device)
        targets_policy = torch.from_numpy(batch.policy_indices).to(self.device)
        target_value_outcome = torch.from_numpy(batch.value_outcomes).to(self.device)  # one-hot
        target_value_mcts = torch.from_numpy(batch.value_mcts).to(self.device)         # dense
        value_labels = torch.argmax(target_value_outcome, dim=-1)
        value_q_labels = torch.argmax(target_value_mcts, dim=-1)

        with amp.autocast(device_type=self.device_type, dtype=self.amp_dtype):
            policy_logits, value_logits, value_q_logits = self.model(
                [inputs, None], return_all=True
            )
            B = policy_logits.shape[0]
            policy_logits = policy_logits.view(B, -1)
            value_logits = value_logits.view(B, 3)
            value_q_logits = value_q_logits.view(B, 3)

            policy_loss = F.cross_entropy(
                policy_logits, targets_policy, ignore_index=NULL_POLICY_INDEX
            )

            # value head ← sparse one-hot game-outcome WDL (cross-entropy on soft labels).
            value_loss = F.cross_entropy(value_logits, target_value_outcome)
            # value_q head ← dense MCTS-backed-up WDL (cross-entropy on soft labels).
            value_q_loss = F.cross_entropy(value_q_logits, target_value_mcts)

            value_accuracy = (
                (value_logits.argmax(dim=-1) == value_labels).float().mean()
            )
            value_q_accuracy = (
                (value_q_logits.argmax(dim=-1) == value_q_labels).float().mean()
            )

            total_loss = 0.1 * policy_loss + value_q_loss + value_loss

        self.scaler.scale(total_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Keep teacher in sync every step using EMA
        self._update_teacher()
        policy_accuracy = (
            (policy_logits.argmax(dim=-1) == targets_policy)
            .float()
            .mean()
            .item()
        )

        # Calculate policy entropy: -sum(p * log(p + eps))
        # Convert to float32 for accurate entropy calculation
        policy_logits_f32 = policy_logits.float()
        policy_probs = F.softmax(policy_logits_f32, dim=-1)
        eps = 1e-8
        policy_entropy = -(policy_probs * torch.log(policy_probs + eps)).sum(dim=-1).mean().item()

        # Calculate probability mass on legal moves
        # Compute softmax of logits, multiply with legal moves mask, then sum
        legal_move_probs = self._compute_legal_move_probability(
            policy_probs, batch.fens
        )

        # Buffer-target sharpness: what fraction of MCTS-WDL targets are
        # confident (>=80% on a single class)? When this stays low the value_q
        # head has nothing to learn from. Cheap — computed on the same batch.
        vm = target_value_mcts.float()
        frac_w_conf = float((vm[:, 0] >= 0.8).float().mean().item())
        frac_l_conf = float((vm[:, 2] >= 0.8).float().mean().item())
        frac_d_conf = float((vm[:, 1] >= 0.8).float().mean().item())
        frac_uniform = float((vm.max(dim=-1).values < 0.40).float().mean().item())

        metrics = {
            "loss/total": float(total_loss.detach().item()),
            "loss/policy": float(policy_loss.detach().item()),
            "loss/value": float(value_loss.detach().item()),
            "loss/value_q": float(value_q_loss.detach().item()),
            "acc/policy": policy_accuracy,
            "acc/value": float(value_accuracy.detach().item()),
            "acc/value_q": float(value_q_accuracy.detach().item()),
            "entropy/policy": policy_entropy,
            "legal_move_prob": legal_move_probs,
            "buffer/frac_W_geq_0.8": frac_w_conf,
            "buffer/frac_L_geq_0.8": frac_l_conf,
            "buffer/frac_D_geq_0.8": frac_d_conf,
            "buffer/frac_near_uniform": frac_uniform,
        }
        return metrics

    # ------------------------------------------------------------------ #
    def _log(self, metrics: Dict[str, float]) -> None:
        elapsed = time.time() - self.last_log_time
        if elapsed <= 0:
            elapsed = 1e-6
        positions_per_sec = (self.config.batch_size * self.config.log_every) / elapsed
        self.last_log_time = time.time()

        g = metrics.get("pacing/g_records_per_sec", 0.0)
        k = metrics.get("pacing/k_actual", 0.0)
        log_line = (
            f"[step {self.step}] "
            f"loss={metrics['loss/total']:.4f} "
            f"policy_acc={metrics['acc/policy']:.3f} "
            f"value_acc={metrics['acc/value']:.3f} "
            f"pps={positions_per_sec:,.0f} "
            f"g={g:.1f}/s k={k:.2f}"
        )
        self.console.log(log_line)

        if self._wandb_run:
            wandb.log(
                {
                    **metrics,
                    "speed/positions_per_sec": positions_per_sec,
                    "global_step": self.step,
                }
            )

    # ------------------------------------------------------------------ #
    def _update_teacher(self) -> None:
        """Update teacher model parameters using EMA.

        Two pieces of state to keep in sync:

        1. Parameters (conv weights, FC weights, BN gamma/beta, ...) get
           EMA-blended:   teacher = decay * teacher + (1-decay) * student.
        2. Buffers (BN running_mean / running_var / num_batches_tracked) get
           hard-copied from the student. These are not learnable; they track
           the *running statistics* of activations seen during the student's
           training. If we leave them at init while EMA-updating gamma/beta,
           the teacher's BN normalizes using stale stats and the network's
           output is garbage at eval() time. (model.parameters() does NOT
           iterate buffers, which was the original bug.)
        """
        with torch.no_grad():
            for teacher_param, student_param in zip(self.teacher_model.parameters(), self.model.parameters()):
                teacher_param.data.mul_(self.config.teacher_decay).add_(
                    student_param.data, alpha=self.config.teacher_weight
                )
            for teacher_buf, student_buf in zip(self.teacher_model.buffers(), self.model.buffers()):
                teacher_buf.data.copy_(student_buf.data)

    def _save_checkpoint(self) -> None:
        # Save Student Checkpoint (for resuming training)
        ckpt = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "step": self.step,
            "config": self.config.to_dict(),
        }
        path = self.checkpoint_dir / f"step_{self.step:08d}.pt"
        torch.save(ckpt, path)
        self.console.log(f"Saved student checkpoint to {path}")

        # Save Teacher Checkpoint (for self-play)
        teacher_ckpt = {
            "model": self.teacher_model.state_dict(),
            "step": self.step,
            "config": self.config.to_dict(),
        }
        teacher_path = self.checkpoint_dir / f"teacher_step_{self.step:08d}.pt"
        torch.save(teacher_ckpt, teacher_path)
        self.console.log(f"Saved teacher checkpoint to {teacher_path}")

        # Build engine from STUDENT checkpoint
        self._invoke_build_script(path)
        
        # Evaluate STUDENT checkpoint (to see progress)
        # self._evaluate_checkpoint(path)
        
        self._prune_checkpoints()

    def _load_checkpoint(self, path: Path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler"])
        self.step = int(ckpt.get("step", 0))
        self.console.log(f"Resumed from checkpoint {path} at step {self.step}")

    def _prune_checkpoints(self) -> None:
        keep = self.config.checkpoint.keep_last
        if keep <= 0:
            return
        files = sorted(self.checkpoint_dir.glob("step_*.pt"))
        for old in files[:-keep]:
            try:
                old.unlink()
            except OSError:
                pass
        
        # Prune teacher checkpoints
        teacher_files = sorted(self.checkpoint_dir.glob("teacher_step_*.pt"))
        for old in teacher_files[:-keep]:
            try:
                old.unlink()
            except OSError:
                pass

    def _invoke_build_script(self, checkpoint_path: Path) -> None:
        script = self.config.checkpoint.build_script
        if script is None:
            return
        script_path = Path(script).resolve()
        if not script_path.is_file():
            self.console.log(f"[WARN] build script {script_path} not found; skipping TRT export.")
            return

        engine_dir = self.config.checkpoint.engine_dir
        iteration_file = self.config.checkpoint.iteration_file

        cmd = [
            script_path.as_posix(),
            checkpoint_path.resolve().as_posix(),
            str(self.step),
            engine_dir.as_posix(),
            iteration_file.as_posix(),
        ]
        try:
            subprocess.run(cmd, check=True)
            self.console.log(f"Updated TensorRT engine via {script_path}.")
        except Exception as exc:
            self.console.log(f"[WARN] build_trt.sh failed: {exc}")

    def _evaluate_checkpoint(self, checkpoint_path: Path) -> None:
        """Evaluate raw model against Stockfish and log ELO to wandb."""
        self.console.log(f"Evaluating checkpoint {checkpoint_path} against Stockfish (ELO {self._current_stockfish_elo})...")
        
        try:
            from src.eval.play_move import main_model_vs_stockfish
        except ImportError as exc:
            self.console.log(f"[WARN] Failed to import play_move: {exc}")
            return
        
        # Load a fresh model instance from checkpoint
        eval_model = SmallModel().to(self.device)
        eval_model.eval()
        try:
            ckpt = torch.load(checkpoint_path, map_location=self.device,weights_only=False)
            eval_model.load_state_dict(ckpt["model"])
        except Exception as exc:
            self.console.log(f"[WARN] Failed to load checkpoint for evaluation: {exc}")
            return
        
        # Run evaluation
        try:
            win_rate, estimated_elo = main_model_vs_stockfish(
                model=eval_model,
                model1_name="SelfPlay",
                device=self.device.type,
                num_games=100,
                temp=0.1,
                elo=self._current_stockfish_elo,
            )
            
            # Handle case where estimated_elo might be "N/A"
            elo_value = estimated_elo if isinstance(estimated_elo, (int, float)) else None
            
            # Log to wandb
            if self._wandb_run:
                log_dict = {
                    "eval/win_rate": win_rate,
                    "eval/stockfish_elo": self._current_stockfish_elo,
                    "global_step": self.step,
                }
                if elo_value is not None:
                    log_dict["eval/estimated_elo"] = elo_value
                wandb.log(log_dict)
            
            # Handle case where estimated_elo might be "N/A"
            if isinstance(estimated_elo, str):
                self.console.log(f"Evaluation complete: ELO=N/A, Win rate={win_rate:.3f}")
                # Don't update Stockfish ELO if we can't estimate
            else:
                self.console.log(f"Evaluation complete: ELO={estimated_elo:.1f}, Win rate={win_rate:.3f}")
                
                # Update Stockfish ELO for next evaluation
                self._current_stockfish_elo = max(int(estimated_elo), 1400)
                self.console.log(f"Next evaluation will use Stockfish ELO {self._current_stockfish_elo}")
            
        except Exception as exc:
            self.console.log(f"[WARN] Evaluation failed: {exc}")

