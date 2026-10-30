"""The fused CUDA network (cpp/net/encoder_kernel.cu) on a GPU.

Against PyTorch on random planes, and its packed input path (positions as self-play
packs them, expanded inside the kernel) against the dense planes of the CPU encoder:
the two must give identical outputs, so the kernel's plane expansion matches the
reference bit for bit.
"""

from __future__ import annotations

import chess as pchess
import numpy as np
import pytest
import torch

_inference_cpp = pytest.importorskip("_inference_cpp")

from src.export.heads import ExportWrapper  # noqa: E402
from src.export.to_cnet import entries_for, write_cnet  # noqa: E402
from src.models.encoder import EncoderModel  # noqa: E402

pytestmark = pytest.mark.gpu

STARTPOS = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
KIWIPETE = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
CASES = [
    (STARTPOS, []),
    (STARTPOS, ["e2e4"]),
    (KIWIPETE, []),
    (KIWIPETE, ["e1g1", "h3g2", "f3g2"]),  # castling, a capture
    (STARTPOS, ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4"]),
    (STARTPOS, ["g1f3", "g8f6", "f3g1", "f6g8"]),  # repetition
    (STARTPOS, ["e2e4", "a7a6", "e4e5", "d7d5"]),  # en passant available
]


@pytest.fixture(scope="module")
def net(tmp_path_factory):
    torch.manual_seed(0)
    model = EncoderModel().eval()
    with torch.no_grad():  # the trained-only parameters start at zero; make them count
        model.pair_bias.normal_(0, 1.0)
        model.promo.normal_(0, 0.5)
        for blk in model.blocks:
            blk.x0_gain.fill_(0.3)
    path = tmp_path_factory.mktemp("cnet") / "net.cnet"
    write_cnet(path, entries_for(model, "search"))
    return model, str(path)


def test_matches_pytorch(net):
    model, path = net
    rng = np.random.default_rng(0)
    planes = (rng.random((8, 8, 8, 111)) < 0.1).astype(np.float32)
    planes[..., 110] = 0.37
    out = _inference_cpp.fused_forward_dense(path, planes)
    with torch.no_grad():
        ref = [t.numpy() for t in ExportWrapper(model, "search")(torch.from_numpy(planes))]
    assert np.isfinite(out["policy"]).all()
    assert (out["policy"].argmax(1) == ref[0].argmax(1)).all()
    assert np.abs(out["policy"] - ref[0]).max() < 0.05
    assert np.abs(out["value"] - ref[1]).max() < 5e-3
    assert np.abs(out["mlh"] - ref[2]).max() < 0.02
    assert np.abs(out["mlh_q"] - ref[3]).max() < 0.02


def random_games(n: int, seed: int) -> list[list[str]]:
    rng = np.random.default_rng(seed)
    games = []
    for _ in range(n):
        board, moves = pchess.Board(), []
        for _ in range(int(rng.integers(0, 80))):
            legal = list(board.legal_moves)
            if not legal:
                break
            mv = legal[int(rng.integers(0, len(legal)))]
            moves.append(mv.uci())
            board.push(mv)
        games.append(moves)
    return games


def test_packed_input_matches_dense(net):
    _, path = net
    cases = CASES + [(STARTPOS, moves) for moves in random_games(40, 20261030)]
    fens = [fen for fen, _ in cases]
    moves = [mv for _, mv in cases]
    dense = np.stack(
        [
            np.asarray(_inference_cpp.encode_position_with_history(f, m), np.float32)
            for f, m in cases
        ]
    )
    a = _inference_cpp.fused_forward_dense(path, dense.reshape(-1, 8, 8, 111))
    b = _inference_cpp.fused_forward_positions(path, fens, moves)
    for key in a:
        assert np.array_equal(a[key], b[key]), key
