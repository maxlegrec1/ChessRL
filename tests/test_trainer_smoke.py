"""CPU smoke test: train steps on a synthetic v2 buffer, growth, checkpoint + resume."""

import pytest
import torch

from src.train.config import TrainingConfig
from src.train.trainer import Trainer

from .test_dataset import synth_records


def _config(tmp_path, policy_target: str) -> TrainingConfig:
    run = tmp_path / "run"
    (run / "buffer").mkdir(parents=True)
    rows = synth_records(n_games=8)
    rows.tofile(run / "buffer" / "buffer.bin")
    cfg = TrainingConfig()
    cfg.device = "cpu"
    cfg.amp_dtype = "bfloat16"
    cfg.batch_size = 32
    cfg.learning_rate = 1e-3
    cfg.policy_target = policy_target
    cfg.wandb.enabled = False
    cfg.data.dataset_path = run / "buffer" / "buffer.bin"
    cfg.data.min_samples_to_start = 1
    cfg.checkpoint.dir = run / "checkpoints"
    cfg.checkpoint.build_script = None
    cfg.checkpoint.engine_dir = run / "engines"
    cfg.checkpoint.iteration_file = run / "engines" / ".iteration"
    cfg.growth.schedule = [[2, 5, 80]]
    cfg.lr_schedule.kind = "cosine"
    cfg.lr_schedule.total_steps = 10
    cfg.lr_schedule.warmup_steps = 2
    return cfg


@pytest.mark.parametrize("policy_target", ["visits", "played"])
def test_train_grow_checkpoint_resume(tmp_path, policy_target):
    cfg = _config(tmp_path, policy_target)
    trainer = Trainer(cfg)
    losses = []
    for _ in range(4):
        trainer._maybe_grow()
        batch = trainer.buffer.sample_batch(cfg.batch_size, mirror_prob=0.5)
        metrics = trainer._train_step(batch, expensive=True)
        losses.append(float(metrics["loss/total"]))
        trainer.step += 1
    assert all(torch.isfinite(torch.tensor(losses)))
    assert trainer.model.spec() == {"arch": "resnet", "blocks": 5, "channels": 80}
    assert 0.0 <= metrics["legal_move_prob"] <= 1.0
    trainer._save_checkpoint()
    ckpt_path = sorted(cfg.checkpoint.dir.glob("step_*.pt"))[-1]

    cfg.resume_path = ckpt_path
    resumed = Trainer(cfg)
    assert resumed.step == 4
    assert resumed.model.spec() == {"arch": "resnet", "blocks": 5, "channels": 80}
    for a, b in zip(trainer.model.state_dict().values(), resumed.model.state_dict().values(), strict=True):
        torch.testing.assert_close(a, b)
    assert (cfg.checkpoint.dir.parent / "metrics" / "train.jsonl").exists()


def test_lr_schedule_counts_from_resume_step(tmp_path):
    cfg = _config(tmp_path, "visits")
    cfg.growth.schedule = []
    cfg.lr_schedule.warmup_steps = 0
    cfg.lr_schedule.start_step = -1
    trainer = Trainer(cfg)  # fresh run: resolves to step 0
    assert cfg.lr_schedule.start_step == 0
    cfg.lr_schedule.start_step = 100
    assert trainer.lr_at(100) == pytest.approx(cfg.learning_rate)
    assert trainer.lr_at(110) == pytest.approx(cfg.learning_rate * cfg.lr_schedule.final_ratio)
    assert trainer.lr_at(50) == pytest.approx(cfg.learning_rate)  # before the start: no decay yet


def test_lr_schedule_cosine_and_growth_rewarmup(tmp_path):
    cfg = _config(tmp_path, "visits")
    trainer = Trainer(cfg)
    assert trainer.lr_at(0) == pytest.approx(cfg.learning_rate * 0.5)  # warmup 1/2
    assert trainer.lr_at(10) == pytest.approx(cfg.learning_rate * cfg.lr_schedule.final_ratio)
    trainer.last_growth_step = 5
    assert trainer.lr_at(5) < trainer.lr_at(9)


def test_weight_decay_spares_norms_and_biases(tmp_path):
    cfg = _config(tmp_path, "visits")
    cfg.optimizer = "adamw"
    cfg.weight_decay = 0.05
    t = Trainer(cfg)
    decayed, free = t.optimizer.param_groups
    assert decayed["weight_decay"] == 0.05 and free["weight_decay"] == 0.0
    assert all(p.ndim >= 2 for p in decayed["params"]) and all(p.ndim < 2 for p in free["params"])
    assert len(decayed["params"]) + len(free["params"]) == len(list(t.model.parameters()))
