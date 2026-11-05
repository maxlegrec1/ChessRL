"""The wide networks' path (cpp/net/encoder_wide.cu: networks the fused kernel cannot hold) on a GPU.

Against PyTorch on random planes, at heads of 64 and 32 columns, with both value-head mixes; its packed input path
(positions as self-play packs them, expanded on the GPU as the fused kernel does) against the dense planes of the CPU
encoder, which must give identical outputs; and the exporter's refusal of INT8 for these sizes.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

_inference_cpp = pytest.importorskip("_inference_cpp")

from src.export.heads import ExportWrapper  # noqa: E402
from src.export.to_cnet import entries_for, write_cnet  # noqa: E402
from src.models.encoder import EncoderModel  # noqa: E402
from tests.test_fused_network import CASES, STARTPOS, random_games  # noqa: E402

pytestmark = pytest.mark.gpu

SIZES = {"hd64": (256, 4, 512), "hd32": (192, 6, 256)}  # d_model, heads, d_ff


def wide_model(d: int, heads: int, d_ff: int) -> EncoderModel:
    torch.manual_seed(0)
    model = EncoderModel(d_model=d, layers=2, heads=heads, d_ff=d_ff).eval()
    with torch.no_grad():  # the trained-only parameters start at zero; make them count
        model.pair_bias.normal_(0, 1.0)
        model.promo.normal_(0, 0.5)
        for blk in model.blocks:
            blk.x0_gain.fill_(0.3)
    return model


@pytest.fixture(scope="module", params=sorted(SIZES))
def net(request, tmp_path_factory):
    model = wide_model(*SIZES[request.param])
    path = tmp_path_factory.mktemp("cnet") / f"{request.param}.cnet"
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


def test_blend_heads(tmp_path):
    """search_heads blend: the WDL mixes both value heads' softmaxes (wdl_w), mlh_q repeats mlh."""
    model = wide_model(*SIZES["hd64"])
    path = tmp_path / "blend.cnet"
    write_cnet(path, entries_for(model, "blend", q_weight=0.3))
    rng = np.random.default_rng(1)
    planes = (rng.random((4, 8, 8, 111)) < 0.1).astype(np.float32)
    out = _inference_cpp.fused_forward_dense(str(path), planes)
    with torch.no_grad():
        ref = [t.numpy() for t in ExportWrapper(model, "blend", 0.3)(torch.from_numpy(planes))]
    assert np.abs(out["value"] - ref[1]).max() < 5e-3
    assert np.array_equal(out["mlh_q"], out["mlh"])


def test_packed_input_matches_dense(net):
    _, path = net
    cases = CASES + [(STARTPOS, moves) for moves in random_games(40, 20261105)]
    fens = [fen for fen, _ in cases]
    moves = [mv for _, mv in cases]
    dense = np.stack(
        [np.asarray(_inference_cpp.encode_position_with_history(f, m), np.float32) for f, m in cases]
    )
    a = _inference_cpp.fused_forward_dense(path, dense.reshape(-1, 8, 8, 111))
    b = _inference_cpp.fused_forward_positions(path, fens, moves)
    for key in a:
        assert np.array_equal(a[key], b[key]), key


def test_int8_export_refused():
    with pytest.raises(ValueError, match="INT8"):
        entries_for(wide_model(*SIZES["hd64"]), "search", int8=True)
