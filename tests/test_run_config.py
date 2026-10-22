"""Launcher config resolution: resume keeps the run's own settings."""

import yaml

from src import run as launcher


def test_resume_without_config_keeps_run_settings(tmp_path, monkeypatch):
    run_dir = tmp_path / "20261021_000000_abcd_x"
    for sub in launcher.SUBDIRS:
        (run_dir / sub).mkdir(parents=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump({"policy_target": "visits", "selfplay": {"pcr_full_prob": 0.25}}))
    out = launcher._write_run_config(run_dir / "config.yaml", run_dir, resume_from=None)
    cfg = yaml.safe_load(out.read_text())
    assert cfg["policy_target"] == "visits"
    assert cfg["selfplay"]["pcr_full_prob"] == 0.25
    assert cfg["data"]["dataset_path"].endswith("buffer/buffer.bin")


def test_extends_resolved_into_snapshot(tmp_path):
    (tmp_path / "base.yaml").write_text("batch_size: 64\nselfplay:\n  simulations: 50\n")
    (tmp_path / "arm.yaml").write_text("extends: base.yaml\nselfplay:\n  cpuct: 1.0\n")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    out = launcher._write_run_config(tmp_path / "arm.yaml", run_dir, resume_from=None)
    cfg = yaml.safe_load(out.read_text())
    assert cfg["batch_size"] == 64 and cfg["selfplay"]["simulations"] == 50 and cfg["selfplay"]["cpuct"] == 1.0
    assert "extends" not in cfg
