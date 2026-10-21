from __future__ import annotations

import json
import math
import subprocess
import time
from collections import deque
from pathlib import Path

import chess
import numpy as np
import torch
import torch.nn.functional as F
from rich.console import Console
from torch import amp, optim

from src.models.factory import build_model, normalize_spec
from src.models.growth import grow_resnet
from src.models.small.vocab import policy_index, policy_to_idx

from .config import TrainingConfig
from .dataset import (
    CASTLE_INDEX,
    FLAG_FULL_SEARCH,
    IGNORE_INDEX,
    RECORD_SIZE,
    Batch,
    Prefetcher,
    ReplayBuffer,
    decode_planes,
    dense_policy_target,
)

AMP_DTYPES: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}

_CASTLE_IDX = torch.tensor(sorted(CASTLE_INDEX))
_E1, _E8 = 4, 60


def legal_move_probability(probs: torch.Tensor, fens: list[str]) -> float:
    """Mean policy mass on legal moves (python-chess; call on a handful of rows)."""
    total = 0.0
    for p, fen in zip(probs, fens, strict=True):
        board = chess.Board(fen)
        idx = []
        for move in board.legal_moves:
            uci = move.uci()
            uci = uci[:-1] if uci.endswith("n") else uci
            if uci in policy_to_idx:
                idx.append(policy_to_idx[uci])
        total += float(p[idx].sum()) if idx else 0.0
    return total / max(1, len(fens))


class Trainer:
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.console = Console()
        self.device = torch.device(config.device)
        self.device_type = config.device
        self.run_dir = config.checkpoint.dir.parent
        self.metrics_path = config.metrics_path or (self.run_dir / "metrics" / "train.jsonl")
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)

        self.buffer = ReplayBuffer(
            config.data.dataset_path,
            window_c=config.data.window_c,
            window_alpha=config.data.window_alpha,
            window_beta=config.data.window_beta,
        )

        self.step = 0
        self.samples_processed = 0
        self.last_growth_step = -(10**9)
        self.amp_dtype = AMP_DTYPES.get(config.amp_dtype.lower(), torch.float16)
        self.scaler = amp.GradScaler(device=self.device_type, enabled=self.amp_dtype == torch.float16)

        spec = normalize_spec(config.model)
        ckpt = None
        if config.resume_path:
            ckpt = torch.load(config.resume_path, map_location=self.device, weights_only=False)
            spec = normalize_spec(ckpt.get("model_spec"))
        self.model = build_model(spec).to(self.device)
        self.model.train()
        self.optimizer = self._make_optimizer()
        self._forward = torch.compile(self.model) if config.compile else self.model

        self.checkpoint_dir = config.checkpoint.dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._build_proc: subprocess.Popen | None = None
        self._build_log = None

        self._wandb_run = self._init_wandb()
        if ckpt is not None:
            self._load_checkpoint(ckpt, config.resume_path)
        self._log_event("start", spec=self.model_spec(), params=self.param_count())

    # ------------------------------------------------------------------ #
    def model_spec(self) -> dict:
        return self.model.spec()

    def param_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def _make_optimizer(self) -> optim.Optimizer:
        cfg = self.config
        kwargs = dict(lr=cfg.learning_rate, weight_decay=cfg.weight_decay, betas=(cfg.beta1, cfg.beta2))
        if cfg.optimizer == "adamw":
            return optim.AdamW(self.model.parameters(), **kwargs)
        return optim.Adam(self.model.parameters(), **kwargs)

    def lr_at(self, step: int) -> float:
        cfg = self.config
        sched = cfg.lr_schedule
        factor = 1.0
        if sched.kind == "cosine" and sched.total_steps > 0:
            progress = min(1.0, step / sched.total_steps)
            factor = sched.final_ratio + (1.0 - sched.final_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
        elif sched.kind == "steps":
            for drop_step, drop_factor in sorted(sched.drops):
                if step >= drop_step:
                    factor = float(drop_factor)
        warm = 1.0
        if sched.warmup_steps > 0:
            warm = min(1.0, (step + 1) / sched.warmup_steps)
        rewarm = cfg.growth.lr_rewarmup_steps
        if rewarm > 0 and step - self.last_growth_step < rewarm:
            warm = min(warm, (step - self.last_growth_step + 1) / rewarm)
        return cfg.learning_rate * factor * warm

    # ------------------------------------------------------------------ #
    def _init_wandb(self):
        if not self.config.wandb.enabled:
            return None
        import wandb

        wandb_dir = self.config.wandb.dir
        if wandb_dir is not None:
            Path(wandb_dir).mkdir(parents=True, exist_ok=True)
        init_kwargs = dict(
            project=self.config.wandb.project,
            entity=self.config.wandb.entity,
            name=self.config.wandb.run_name or self.config.run_name,
            config=self.config.to_dict(),
            dir=str(wandb_dir) if wandb_dir is not None else None,
        )
        if self.config.wandb.resume_id:
            init_kwargs["id"] = self.config.wandb.resume_id
            init_kwargs["resume"] = "allow"
        run = wandb.init(**init_kwargs)
        # Surface the run id so external metric producers can attach.
        try:
            (self.run_dir / "wandb_run_id.txt").write_text(f"{run.id}\n")
        except Exception as exc:  # never fail training over this
            self.console.log(f"[WARN] failed to write wandb_run_id.txt: {exc}")
        return run

    def _write_metrics(self, row: dict) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        if self._wandb_run:
            import wandb

            wandb.log({k: v for k, v in row.items() if isinstance(v, (int, float))})

    def _log_event(self, event: str, **fields) -> None:
        self._write_metrics({"event": event, "step": self.step, "t": time.time(), **fields})
        self.console.log(f"[event] {event} {fields}")

    # ------------------------------------------------------------------ #
    def _wait_for_warmup(self) -> None:
        threshold = self.config.data.min_samples_to_start
        poll = max(1.0, float(self.config.data.warmup_poll_seconds))
        path = self.config.data.dataset_path

        def live_size() -> int:
            try:
                return path.stat().st_size // RECORD_SIZE
            except FileNotFoundError:
                return 0

        if live_size() < threshold:
            self.console.log(f"[warmup] waiting for buffer to reach {threshold:,} samples")
        while live_size() < threshold:
            time.sleep(poll)
        self.buffer.refresh()
        self.console.log(f"[warmup] buffer at {self.buffer.size:,}; starting training")

    def _maybe_grow(self) -> None:
        if self.model.spec()["arch"] != "resnet":
            return
        for stage in sorted(self.config.growth.schedule):
            at_step, blocks, channels = (int(v) for v in stage)
            cur = self.model.spec()
            if self.step < at_step or (blocks <= cur["blocks"] and channels <= cur["channels"]):
                continue
            blocks, channels = max(blocks, cur["blocks"]), max(channels, cur["channels"])
            before = self.param_count()
            self.model = grow_resnet(self.model, blocks, channels)
            self.model.train()
            self.optimizer = self._make_optimizer()
            self._forward = torch.compile(self.model) if self.config.compile else self.model
            self.last_growth_step = self.step
            self._log_event("grow", spec=self.model.spec(), params_before=before, params=self.param_count())

    def run(self) -> None:
        self._wait_for_warmup()
        cfg = self.config
        target_k = max(0.1, float(cfg.data.target_k))
        prefetch = Prefetcher(self.buffer, cfg.batch_size, cfg.mirror_prob)
        last_refresh = time.monotonic()

        # Sliding-window measurement of the self-play insertion rate g from the
        # monotonic write counter. Records arrive in game-end bursts, so a 30 s
        # window absorbs them; total_writes keeps moving once the buffer is full.
        g_window_seconds = 30.0
        g_history: deque = deque()
        g_history.append((time.time(), self.buffer.total_writes))
        step_times: deque = deque(maxlen=50)

        try:
            while True:
                if time.monotonic() - last_refresh >= cfg.data.reload_seconds:
                    prefetch.refresh()
                    last_refresh = time.monotonic()
                self._maybe_grow()

                t0 = time.time()
                batch = prefetch.get().to(self.device)
                expensive = self.step % max(1, cfg.expensive_metrics_every) == 0
                metrics = self._train_step(batch, expensive)
                t1 = time.time()
                step_times.append(t1 - t0)

                cur_writes = self.buffer.total_writes
                g_history.append((t1, cur_writes))
                while len(g_history) > 2 and (t1 - g_history[0][0]) > max(g_window_seconds * 4, 300.0):
                    g_history.popleft()
                anchor_t, anchor_w = g_history[0]
                for ts, w in g_history:
                    if t1 - ts < g_window_seconds:
                        break
                    anchor_t, anchor_w = ts, w
                window_dt = max(1e-3, t1 - anchor_t)
                g = max(0.0, (cur_writes - anchor_w) / window_dt)
                # Until the window fills, cap g so one burst doesn't make the trainer sprint.
                g_for_pacing = min(g, 50.0) if window_dt < g_window_seconds * 0.5 else g
                target_step_interval = cfg.batch_size / max(1e-3, target_k * max(g_for_pacing, 1.0))
                step_dt = t1 - t0
                sleep_for = max(0.0, target_step_interval - step_dt)
                time.sleep(sleep_for)

                self.step += 1
                self.samples_processed += cfg.batch_size

                if self.step % cfg.log_every == 0:
                    eff = max(step_dt, target_step_interval)
                    row = {k: float(v) for k, v in metrics.items()}
                    row.update(
                        step=self.step,
                        t=time.time(),
                        samples=self.samples_processed,
                        lr=self.lr_at(self.step),
                        params=self.param_count(),
                        **{f"model_{k}": v for k, v in self.model.spec().items() if k != "arch"},
                        **{
                            "pacing/g_records_per_sec": g,
                            "pacing/s_records_per_sec": cfg.batch_size / max(1e-6, eff),
                            "pacing/k_actual": (cfg.batch_size / max(1e-6, eff)) / max(1e-6, g),
                            "pacing/sleep_s": sleep_for,
                            "pacing/step_time_s": float(np.mean(step_times)),
                            "buffer/window": self.buffer.current_window(),
                            "buffer/records": self.buffer.size,
                            "buffer/total_writes": self.buffer.total_writes,
                        },
                    )
                    self._write_metrics(row)
                    if self.step % (cfg.log_every * 10) == 0:
                        self.console.log(
                            f"[step {self.step}] loss={row['loss/total']:.4f} "
                            f"pol={row['loss/policy']:.3f} val={row['loss/value']:.3f} "
                            f"acc={row['acc/policy']:.3f} g={g:.0f}/s k={row['pacing/k_actual']:.2f} "
                            f"win={row['buffer/window']:,} lr={row['lr']:.2e}"
                        )

                if self.step % cfg.checkpoint.every_steps == 0:
                    self._save_checkpoint()
        except KeyboardInterrupt:
            self.console.log("Training interrupted by user.")
        finally:
            prefetch.close()
            if self._wandb_run:
                import wandb

                wandb.finish()

    # ------------------------------------------------------------------ #
    def _train_step(self, batch: Batch, expensive: bool) -> dict[str, torch.Tensor | float]:
        cfg = self.config
        lr = self.lr_at(self.step)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.optimizer.zero_grad(set_to_none=True)

        inputs = decode_planes(batch)
        full = (batch.flags & FLAG_FULL_SEARCH) > 0
        target_mlh = (batch.plies_remaining.clamp_min(0.0) / 25.0).clamp_max(4.0)
        target_mlh_q = (batch.backed_plies.clamp_min(0.0) / 25.0).clamp_max(4.0)
        mlh_mask = (batch.plies_remaining > -0.5).float()
        mlh_q_mask = (batch.backed_plies > -0.5).float()
        has_visits = full & (batch.policy_prob.sum(dim=1) > 0)

        with amp.autocast(device_type=self.device_type, dtype=self.amp_dtype):
            policy_logits, value_logits, value_q_logits, mlh_pred, mlh_q_pred = self._forward(
                inputs, return_all=True
            )
        policy_logits = policy_logits.float()
        value_logits = value_logits.float()
        value_q_logits = value_q_logits.float()
        mlh_pred = mlh_pred.float().view(-1)
        mlh_q_pred = mlh_q_pred.float().view(-1)

        log_probs = F.log_softmax(policy_logits, dim=-1)
        visit_target = dense_policy_target(batch)
        visit_ce = -(visit_target * log_probs).sum(dim=-1)
        visit_loss = (visit_ce * has_visits).sum() / has_visits.sum().clamp_min(1)
        played = torch.where(full, batch.played_idx, torch.full_like(batch.played_idx, IGNORE_INDEX))
        played_loss = F.cross_entropy(policy_logits, played, ignore_index=IGNORE_INDEX)
        policy_loss = visit_loss if cfg.policy_target == "visits" else played_loss

        value_loss = F.cross_entropy(value_logits, batch.value_outcome)
        value_q_loss = F.cross_entropy(value_q_logits, batch.value_mcts)
        mlh_loss = ((mlh_pred - target_mlh).pow(2) * mlh_mask).sum() / mlh_mask.sum().clamp_min(1.0)
        mlh_q_loss = ((mlh_q_pred - target_mlh_q).pow(2) * mlh_q_mask).sum() / mlh_q_mask.sum().clamp_min(1.0)

        lw = cfg.loss_weights
        total_loss = (
            lw.policy * policy_loss
            + lw.value * value_loss
            + lw.value_q * value_q_loss
            + lw.mlh * mlh_loss
            + lw.mlh_q * mlh_q_loss
        )

        self.scaler.scale(total_loss).backward()
        if cfg.grad_clip > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        with torch.no_grad():
            probs = log_probs.exp()
            value_labels = batch.value_outcome.argmax(dim=-1)
            value_q_labels = batch.value_mcts.argmax(dim=-1)
            played_ok = played != IGNORE_INDEX
            top1 = policy_logits.argmax(dim=-1)
            target_entropy = -(visit_target * visit_target.clamp_min(1e-12).log()).sum(-1)
            vm = batch.value_mcts
            castle_cand = torch.isin(batch.played_idx, _CASTLE_IDX.to(self.device))
            kings_home = (((batch.cur_bb[:, 5] >> _E1) & 1) > 0) | (((batch.cur_bb[:, 11] >> _E8) & 1) > 0)
            metrics: dict[str, torch.Tensor | float] = {
                "loss/total": total_loss.detach(),
                "loss/policy": policy_loss.detach(),
                "loss/policy_visits": visit_loss.detach(),
                "loss/policy_played": played_loss.detach(),
                "loss/value": value_loss.detach(),
                "loss/value_q": value_q_loss.detach(),
                "loss/mlh": mlh_loss.detach(),
                "loss/mlh_q": mlh_q_loss.detach(),
                # KL(visits || policy) = CE - H(target): 0 when the net matches the search.
                "policy/kl_to_visits": ((visit_ce - target_entropy) * has_visits).sum()
                / has_visits.sum().clamp_min(1),
                "acc/policy": ((top1 == played) & played_ok).float().sum() / played_ok.float().sum().clamp_min(1),
                "acc/policy_top_visit": (
                    (top1 == visit_target.argmax(dim=-1)) & has_visits
                ).float().sum() / has_visits.float().sum().clamp_min(1),
                "acc/value": (value_logits.argmax(dim=-1) == value_labels).float().mean(),
                "acc/value_q": (value_q_logits.argmax(dim=-1) == value_q_labels).float().mean(),
                "entropy/policy": -(probs * log_probs).sum(dim=-1).mean(),
                "entropy/visit_target": (target_entropy * has_visits).sum() / has_visits.sum().clamp_min(1),
                "buffer/frac_W_geq_0.8": (vm[:, 0] >= 0.8).float().mean(),
                "buffer/frac_D_geq_0.8": (vm[:, 1] >= 0.8).float().mean(),
                "buffer/frac_L_geq_0.8": (vm[:, 2] >= 0.8).float().mean(),
                "buffer/frac_near_uniform": (vm.max(dim=-1).values < 0.40).float().mean(),
                "buffer/decisive_outcome": (value_labels != 1).float().mean(),
                "buffer/wdl_agreement": (value_labels == value_q_labels).float().mean(),
                "buffer/castle_rate": (castle_cand & kings_home).float().mean(),
                "mlh/pred_plies": (mlh_pred * mlh_mask).sum() / mlh_mask.sum().clamp_min(1.0) * 25.0,
                "mlh/target_plies": (target_mlh * mlh_mask).sum() / mlh_mask.sum().clamp_min(1.0) * 25.0,
                "amp/scale": self.scaler.get_scale() if self.scaler.is_enabled() else 1.0,
            }
            if expensive:
                rows = min(16, probs.shape[0])
                metrics["legal_move_prob"] = legal_move_probability(probs[:rows].float().cpu(), batch.fens[:rows])
        return metrics

    # ------------------------------------------------------------------ #
    def _save_checkpoint(self) -> None:
        ckpt = {
            "model": self.model.state_dict(),
            "model_spec": self.model.spec(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "step": self.step,
            "samples": self.samples_processed,
            "last_growth_step": self.last_growth_step,
            "config": self.config.to_dict(),
        }
        path = self.checkpoint_dir / f"step_{self.step:08d}.pt"
        tmp = path.with_suffix(".pt.tmp")
        torch.save(ckpt, tmp)
        tmp.replace(path)
        self.console.log(f"Saved checkpoint {path.name}")
        self._invoke_build_script(path)
        self._prune_checkpoints()

    def _load_checkpoint(self, ckpt: dict, path: Path) -> None:
        missing, unexpected = self.model.load_state_dict(ckpt["model"], strict=False)
        if missing:
            self.console.log(f"[resume] new params not in checkpoint (initialized fresh): {missing}")
        if unexpected:
            self.console.log(f"[resume] checkpoint has unused params: {unexpected}")
        try:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        except (ValueError, KeyError) as exc:
            self.console.log(f"[resume] optimizer state incompatible ({exc}); reinitialized")
        if "scaler" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler"])
        self.step = int(ckpt.get("step", 0))
        self.samples_processed = int(ckpt.get("samples", self.step * self.config.batch_size))
        self.last_growth_step = int(ckpt.get("last_growth_step", self.last_growth_step))
        self.console.log(f"Resumed from checkpoint {path} at step {self.step}")

    def _prune_checkpoints(self) -> None:
        keep = self.config.checkpoint.keep_last
        if keep <= 0:
            return
        for old in sorted(self.checkpoint_dir.glob("step_*.pt"))[:-keep]:
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
        cmd = [
            script_path.as_posix(),
            checkpoint_path.resolve().as_posix(),
            str(self.step),
            self.config.checkpoint.engine_dir.as_posix(),
            self.config.checkpoint.iteration_file.as_posix(),
        ]
        if not self.config.async_engine_build:
            try:
                subprocess.run(cmd, check=True)
                self.console.log(f"Updated TensorRT engine via {script_path}.")
            except Exception as exc:
                self.console.log(f"[WARN] build_trt.sh failed: {exc}")
            return
        if self._build_proc is not None and self._build_proc.poll() is None:
            self._log_event("build_skipped", checkpoint=checkpoint_path.name)
            return
        if self._build_proc is not None and self._build_proc.returncode not in (0, None):
            self.console.log(f"[WARN] previous engine build exited rc={self._build_proc.returncode}")
        log_path = self.run_dir / "engine_builds.log"
        self._build_log = log_path.open("a")
        self._build_proc = subprocess.Popen(cmd, stdout=self._build_log, stderr=subprocess.STDOUT)
        self.console.log(f"Engine build started for {checkpoint_path.name} (pid {self._build_proc.pid})")


__all__ = ["Trainer", "legal_move_probability", "policy_index"]
