"""Plateau-triggered growth, LR hold-then-decay, resignation calibration."""

import json
import time

import pytest

from src.self_play.calibration import calibrate_resign
from src.train.trainer import Trainer

from .test_trainer_smoke import _config


def _ladder(trainer, elos, start_step=0, hours=3.0):
    path = trainer.run_dir / "ladder" / "ladder.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time() - hours * 3600
    rows = [{"t": t0 + i * hours * 3600 / (len(elos) - 1), "step": start_step + 100 * i, "elo": e}
            for i, e in enumerate(elos)]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


@pytest.mark.parametrize("elos,expect", [([900, 1000, 1100, 1200, 1300, 1400, 1500], False),
                                         ([1380, 1400, 1390, 1410, 1395, 1405, 1400], True)])
def test_plateau_detection(tmp_path, elos, expect):
    cfg = _config(tmp_path, "visits")
    cfg.growth.trigger = "plateau"
    cfg.growth.schedule = [[0, 5, 80]]
    t = Trainer(cfg)
    _ladder(t, elos)
    assert t.plateaued() is expect
    t.step = 100
    t._maybe_grow()
    assert (t.model.spec()["blocks"] == 5) is expect


def test_plateau_ignores_ratings_before_last_growth(tmp_path):
    cfg = _config(tmp_path, "visits")
    cfg.growth.trigger = "plateau"
    t = Trainer(cfg)
    _ladder(t, [1400] * 7, start_step=0)
    t.last_growth_step = 10_000  # all ratings predate the growth
    assert not t.plateaued()


def test_lr_hold_then_cosine(tmp_path):
    cfg = _config(tmp_path, "visits")
    cfg.growth.schedule = []
    cfg.lr_schedule.warmup_steps = 0
    cfg.lr_schedule.start_step = 100
    cfg.lr_schedule.hold_steps = 50
    cfg.lr_schedule.total_steps = 150
    t = Trainer(cfg)
    assert t.lr_at(140) == pytest.approx(cfg.learning_rate)  # inside the hold
    assert t.lr_at(150) == pytest.approx(cfg.learning_rate)  # decay starts here
    assert t.lr_at(250) == pytest.approx(cfg.learning_rate * cfg.lr_schedule.final_ratio)
    assert cfg.learning_rate * cfg.lr_schedule.final_ratio < t.lr_at(200) < cfg.learning_rate


def test_resign_calibration():
    assert calibrate_resign(0.05, 400, 30, 0.05) == pytest.approx(0.04)  # 7.5% wrong -> stricter
    assert calibrate_resign(0.05, 400, 4, 0.05) == pytest.approx(0.055)  # 1% wrong -> looser
    assert calibrate_resign(0.05, 400, 16, 0.05) == 0.05  # 4%: within band
    assert calibrate_resign(0.006, 400, 100, 0.05) == 0.005  # floor
