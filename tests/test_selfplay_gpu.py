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


def test_fractional_random_opening_plies(tmp_path):
    """random_opening_plies 1.5: one random ply in half the games, two in the others, so the first
    searched (and, at pcr_full_prob 1, recorded) ply is 1 or 2, never 0."""
    import numpy as np

    from src.train.dataset import FLAG_BLACK_TO_MOVE, RECORD_DTYPE

    torch.manual_seed(0)
    net = tmp_path / "net.cnet"
    write_cnet(net, entries_for(EncoderModel().eval(), "outcome"))
    sp = load_config("src/train/config.yaml").selfplay
    sp.pgn_dir, sp.save_pgn = tmp_path / "pgn", False
    sp.boards, sp.threads, sp.simulations = 128, 8, 8
    sp.pcr_full_prob, sp.resign_threshold, sp.max_plies = 1.0, 0.0, 6
    sp.random_opening_plies = 1.5
    buffer = tmp_path / "buffer.bin"
    worker = _inference_cpp.SelfPlayWorker(build_cpp_config(sp, str(net), buffer))
    for _ in range(200):
        worker.step()
        if worker.stats().games >= 256:
            break
    del worker
    rows = np.fromfile(buffer, dtype=RECORD_DTYPE)
    ply = 2 * (rows["fullmove"].astype(np.int64) - 1) + ((rows["flags"] & FLAG_BLACK_TO_MOVE) > 0)
    counts = np.bincount(ply, minlength=3)
    assert counts[0] == 0
    # Every game records ply 2 (its first or second search); only one-random-ply games record ply 1.
    assert 0.3 < counts[1] / counts[2] < 0.7, counts[:4]
