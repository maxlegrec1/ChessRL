"""Resume an encoder run under a continuation config: grows on the first step, LR counts from the resume step."""

import pytest
import torch

from src.train.trainer import Trainer

from .test_trainer_smoke import _config


def test_encoder_continuation_grows_and_decays(tmp_path):
    cfg = _config(tmp_path, "visits")
    cfg.model.arch = "encoder"
    cfg.model.d_model, cfg.model.layers, cfg.model.heads, cfg.model.d_ff = 32, 2, 4, 64
    cfg.growth.schedule = []
    cfg.lr_schedule.kind = "constant"
    cfg.lr_schedule.warmup_steps = 0
    t = Trainer(cfg)
    for _ in range(3):
        t._train_step(t.buffer.sample_batch(cfg.batch_size), expensive=False)
        t.step += 1
    t._save_checkpoint()
    ckpt = sorted(cfg.checkpoint.dir.glob("step_*.pt"))[-1]

    cfg.resume_path = ckpt
    cfg.growth.schedule = [[0, 4]]
    cfg.growth.lr_rewarmup_steps = 0
    cfg.lr_schedule.kind = "cosine"
    cfg.lr_schedule.start_step = -1
    cfg.lr_schedule.total_steps = 10
    r = Trainer(cfg)
    assert r.step == 3 and cfg.lr_schedule.start_step == 3
    assert r.model.spec()["layers"] == 2
    x = torch.rand(2, 8, 8, 111)
    r.model.eval()
    before = r.model(x)
    r._maybe_grow()
    r.model.eval()
    assert r.model.spec()["layers"] == 4
    torch.testing.assert_close(r.model(x), before, rtol=1e-5, atol=1e-5)
    assert r.lr_at(3) == pytest.approx(cfg.learning_rate)
    assert r.lr_at(13) == pytest.approx(cfg.learning_rate * cfg.lr_schedule.final_ratio)
    r.model.train()
    m = r._train_step(r.buffer.sample_batch(cfg.batch_size), expensive=False, log=True)
    assert torch.isfinite(torch.as_tensor(float(m["loss/total"])))

    # A later plain resume keeps the continuation's start step (and the grown model).
    r.step += 2
    r._save_checkpoint()
    cfg.resume_path = sorted(cfg.checkpoint.dir.glob("step_*.pt"))[-1]
    cfg.lr_schedule.start_step = -1
    again = Trainer(cfg)
    assert again.step == 5 and cfg.lr_schedule.start_step == 3
    assert again.model.spec()["layers"] == 4
