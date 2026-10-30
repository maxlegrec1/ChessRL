"""The real self-play worker on a GPU, at the recipe's concurrency, with the fused network.

Runs the search threads, the batch runner and its hand-back the way self-play does (many
searches waiting on shared batches), so a stall in that machinery (a lost wake-up, a worker
waiting on a search that waits on it) fails here instead of hanging a run.
"""

from __future__ import annotations

import threading

import pytest
import torch

_inference_cpp = pytest.importorskip("_inference_cpp")

from src.export.to_cnet import entries_for, write_cnet  # noqa: E402
from src.models.encoder import EncoderModel  # noqa: E402
from src.self_play.__main__ import build_cpp_config  # noqa: E402
from src.train.config import load_config  # noqa: E402

pytestmark = pytest.mark.gpu


def test_selfplay_steps_at_recipe_concurrency(tmp_path):
    torch.manual_seed(0)
    net = tmp_path / "net.cnet"
    write_cnet(net, entries_for(EncoderModel().eval(), "outcome"))
    sp = load_config("src/train/config.yaml").selfplay
    sp.pgn_dir = tmp_path / "pgn"
    sp.save_pgn = False
    cfg = build_cpp_config(sp, str(net), tmp_path / "buffer.bin")
    worker = _inference_cpp.SelfPlayWorker(cfg)

    done: list[int] = []

    def play() -> None:
        for _ in range(30):
            done.append(worker.step().positions_generated)

    t = threading.Thread(target=play, daemon=True)
    t.start()
    t.join(timeout=120)
    assert not t.is_alive(), f"self-play stalled after {len(done)} of 30 steps"
    assert sum(done) > 0  # full searches were recorded
