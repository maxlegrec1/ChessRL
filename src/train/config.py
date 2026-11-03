from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class CheckpointConfig:
    dir: Path = Path("artifacts/checkpoints")
    every_steps: int = 1000
    keep_last: int = 5
    build_script: Path | None = Path("scripts/build_net.sh")
    engine_dir: Path = Path("artifacts/engines")
    iteration_file: Path = Path("artifacts/engines/.iteration")


@dataclass
class DataConfig:
    dataset_path: Path = Path("artifacts/buffers/buffer.bin")
    min_samples_to_start: int = 8192
    warmup_poll_seconds: float = 10.0
    # Pacing: the trainer consumes at most target_k samples per record self-play has
    # written since training started (a token bucket on the buffer's write counter).
    # AlphaZero-conservative ≈ 3, Lc0-typical ≈ 5, aggressive ≈ 10+.
    target_k: float = 5.0
    # Recent-data sampling window (KataGo-style growth). Requires a FIFO buffer
    # (selfplay.fifo_buffer). Window over the newest records:
    #   W(N) = c * (1 + beta * ((N / c) ** alpha - 1) / alpha), capped by the buffer,
    # where N = total records ever written. window_c <= 0 samples the whole buffer.
    window_c: int = 0
    window_alpha: float = 0.75
    window_beta: float = 0.4


@dataclass
class RampConfig:
    """A setting moved linearly from `start` to `end` between trainer steps `start_step` and `end_step`
    (`start` before, `end` after). Off while end_step is 0: the setting keeps its own value."""

    start: float = 0.0
    end: float = 0.0
    start_step: int = 0
    end_step: int = 0

    @property
    def on(self) -> bool:
        return self.end_step > 0

    def at(self, step: int) -> float:
        if step <= self.start_step:
            return self.start
        if step >= self.end_step:
            return self.end
        frac = (step - self.start_step) / (self.end_step - self.start_step)
        return self.start + (self.end - self.start) * frac


@dataclass
class SelfPlayConfig:
    # parallelism / batching
    boards: int = 1024
    threads: int = 256
    async_play: bool = False
    hash_size: int = 0
    # MCTS settings, copied by name onto the C++ config (self_play/__main__.py _CPP_FIELDS)
    simulations: int = 800
    cpuct: float = 0.65
    temperature: float = 1.3
    temperature_at_root: float = 1.0
    # Virtual-loss width: 1 = strictly sequential descent; W > 1 collects W leaves per
    # inference round (W-times bigger GPU batches, blurrier search). 8 = throughput knee.
    virtual_loss_width: int = 8
    use_dirichlet: bool = True
    dirichlet_alpha: float = 0.12
    dirichlet_epsilon: float = 0.05
    # buffer / output
    max_buffer_size: int = 10_000_000
    poll_seconds: int = 300
    save_pgn: bool = False
    pgn_dir: Path = Path("artifacts/pgns")
    # Uniform-random legal plies at the start of every game before MCTS takes over
    # (0 = start every game from the initial position). A fraction adds one more ply to that
    # share of games: 2.25 plays 3 random plies in a quarter of the games, 2 in the others.
    random_opening_plies: float = 0.0
    # Ramp random_opening_plies over the networks' trainer steps (self-play applies the value
    # at each network it loads), e.g. from 8 to 0: random openings, then searched ones.
    random_opening_ramp: RampConfig = field(default_factory=RampConfig)
    temperature_plies: int = -1  # -1: temperature_at_root for the whole game
    temperature_late: float = 0.0  # root temperature after temperature_plies (0 = argmax)
    # Root temperature of the fast searches (pcr_fast_simulations) during temperature_plies; below
    # 0 they use temperature_at_root. Their visit counts are flat, so sampling them at 1 blunders.
    fast_temperature_at_root: float = -1.0
    resign_threshold: float = 0.0  # expected score below which the side to move resigns; 0 = off
    resign_consecutive: int = 2
    resign_playthrough: float = 0.1  # fraction of games that never resign (false-positive audit)
    # Auto-calibrate resign_threshold so that, in playthrough games, at most this
    # fraction of would-be resignations were wrong (AlphaZero targeted 5%).
    # 0 = fixed threshold. Measured with a fixed 5% threshold: 7.1-7.4% wrong.
    resign_target_fp: float = 0.0
    resign_calibration_window: int = 400  # would-be resignations per adjustment
    max_plies: int = 600
    pcr_full_prob: float = 1.0  # playout-cap randomization; 1.0 = every move is a full search
    pcr_fast_simulations: int = 100
    # > 0: while moves are sampled (temperature_plies), cheap moves sample the raw
    # policy at this temperature instead of a pcr_fast_simulations search. 0 = off.
    quick_policy_temperature: float = 0.0
    # Keep the played move's subtree for the game's next search, which then runs only the
    # simulations its root is missing.
    tree_reuse: bool = False
    fifo_buffer: bool = False  # ring buffer instead of random replacement when full


@dataclass
class LossWeightsConfig:
    """Per-head weights in the trainer's total loss."""

    policy: float = 0.1
    value: float = 1.0
    value_q: float = 1.0
    mlh: float = 0.05
    mlh_q: float = 0.05


@dataclass
class ModelConfig:
    """Network architecture. Stored in every checkpoint so export can rebuild it."""

    arch: str = "encoder"  # the transformer in src/models/encoder.py, one token per square
    d_model: int = 128
    layers: int = 4
    heads: int = 8
    d_ff: int = 256
    # Input plane 104 (side to move). Positions are canonical (the side to move plays white), so "constant"
    # sets it to 1 everywhere; "colour" sets it to the real side to move before mirroring (1 White, 0 Black),
    # the network's only sign that it plays the black pieces. Saved with the network (model_spec, .cnet).
    stm_plane: str = "constant"


@dataclass
class GrowthConfig:
    """Function-preserving trunk growth during training (src/models/growth.py).

    `schedule` lists stages, [trainer_step, layers]; at each stage the encoder grows to
    that many layers so that the network's output is unchanged at the moment of growth."""

    schedule: list = field(default_factory=list)
    lr_rewarmup_steps: int = 200
    # "steps": grow when a stage's step is reached. "plateau": additionally wait
    # until the run's ladder Elo gained < plateau_min_gain over the last
    # plateau_hours (only ratings since the previous growth count).
    trigger: str = "steps"
    plateau_hours: float = 2.0
    plateau_min_gain: float = 50.0
    plateau_min_ratings: int = 4


@dataclass
class ScheduleConfig:
    """Learning-rate schedule on top of `learning_rate`."""

    kind: str = "constant"  # "constant" | "cosine" | "step"
    # Step the warmup / cosine count from; -1 = the step the run resumes at
    # (resolved once, then saved in checkpoints so later resumes keep it).
    start_step: int = 0
    warmup_steps: int = 0
    total_steps: int = 0  # cosine horizon (including hold_steps)
    hold_steps: int = 0  # cosine: constant LR for this many steps, then decay
    final_ratio: float = 0.1  # cosine floor as a fraction of learning_rate
    # step: the rate is multiplied by gamma at each milestone (steps counted like the cosine's,
    # from start_step), e.g. milestones [70000], gamma 0.5 halves it from step 70,000 on.
    milestones: list = field(default_factory=list)
    gamma: float = 0.5


@dataclass
class TrainingConfig:
    """The defaults are older values, not the recipe: src/train/config.yaml sets the recipe.
    A run's config.yaml snapshot stores only the keys its YAML sets, so changing a default
    here changes what resumed runs get."""

    device: str = "cuda"
    batch_size: int = 256
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    log_every: int = 50
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    data: DataConfig = field(default_factory=DataConfig)
    selfplay: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    loss_weights: LossWeightsConfig = field(default_factory=LossWeightsConfig)
    amp_dtype: str = "float16"
    resume_path: Path | None = None
    model: ModelConfig = field(default_factory=ModelConfig)
    growth: GrowthConfig = field(default_factory=GrowthConfig)
    lr_schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    optimizer: str = "adamw"  # "adamw" | "muon"
    # optimizer: muon (src/train/muon.py): Muon on the encoder layers' weight matrices, AdamW (learning_rate,
    # betas, weight_decay) on the other parameters. The schedule scales both learning rates alike.
    muon_lr: float = 0.006
    muon_momentum: float = 0.95
    muon_weight_decay: float = 0.01
    grad_clip: float = 0.0  # 0 = off
    # "played": one-hot on the move actually played. "visits": MCTS visit
    # distribution stored in the v2 record.
    policy_target: str = "played"
    # Which heads the exported engine feeds MCTS. "search": value_q and mlh_q,
    # trained on the search's own backed-up values; with an untrained net
    # those targets are ~uniform (measured: 98.7% with max prob < 0.4), so the
    # search value only learns from mates found inside the tree. "outcome":
    # value and mlh, trained on game results / true plies left (AlphaZero).
    # "q_value" and "blend" change only the value (see ExportWrapper in src/export/heads.py).
    search_heads: str = "search"
    # search_heads "blend": the weight of value_q (trained on the search's values) in the WDL the
    # search reads, the rest on value (trained on game results). search_q_ramp moves it over
    # trainer steps: each network is exported with the weight at its own step.
    search_q_weight: float = 0.5
    search_q_ramp: RampConfig = field(default_factory=RampConfig)
    compile: bool = False  # torch.compile the training step's forward and losses
    # FP16 accumulators in the FP16 matrix multiplies (torch.backends.cuda.matmul.
    # allow_fp16_accumulation): twice the tensor-core rate on GeForce cards. Neutral at today's
    # size (the multiplies are bound by memory), needed once the model grows.
    fp16_accumulation: bool = False
    # The encoder's residual stream in FP16 during training (src/models/encoder.py FP16_RESIDUAL):
    # half the bytes the step moves through it. The self-play kernel keeps it FP32.
    fp16_residual: bool = False
    async_engine_build: bool = False  # export networks in the background
    # src.run: self-play and trainer under NVIDIA MPS, at the same priority (target_k paces the
    # trainer; below self-play's priority it starved whenever the searches saturated the GPU).
    mps: bool = False
    metrics_path: Path | None = None  # JSONL metrics for the dashboard
    expensive_metrics_every: int = 50  # cadence of host-syncing diagnostics

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge_dataclass(instance, updates: dict[str, Any]):
    for key, value in updates.items():
        if not hasattr(instance, key):
            continue
        attr = getattr(instance, key)
        if isinstance(attr, Path):
            setattr(instance, key, Path(value) if value is not None else None)
        elif hasattr(attr, "__dataclass_fields__") and isinstance(value, dict):
            _merge_dataclass(attr, value)
        else:
            setattr(instance, key, value)


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_yaml(path: Path | str) -> dict:
    """Read a config YAML, resolving `extends: <relative path>` chains (child wins)."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    parent = data.pop("extends", None)
    if parent:
        data = _deep_merge(load_yaml(path.parent / parent), data)
    return data


def load_config(path: Path | str) -> TrainingConfig:
    path = Path(path)
    cfg = TrainingConfig()
    if path.is_file():
        _merge_dataclass(cfg, load_yaml(path))
    cfg.checkpoint.dir = Path(cfg.checkpoint.dir)
    if cfg.checkpoint.build_script is not None:
        cfg.checkpoint.build_script = Path(cfg.checkpoint.build_script)
        if cfg.checkpoint.build_script.name == "build_trt.sh":
            # Configs saved before the fused network replaced TensorRT.
            cfg.checkpoint.build_script = cfg.checkpoint.build_script.with_name("build_net.sh")
    cfg.checkpoint.engine_dir = Path(cfg.checkpoint.engine_dir)
    cfg.checkpoint.iteration_file = Path(cfg.checkpoint.iteration_file)
    cfg.selfplay.pgn_dir = Path(cfg.selfplay.pgn_dir)
    cfg.data.dataset_path = Path(cfg.data.dataset_path)
    if cfg.metrics_path is not None:
        cfg.metrics_path = Path(cfg.metrics_path)
    if cfg.resume_path:
        cfg.resume_path = Path(cfg.resume_path)
    return cfg
