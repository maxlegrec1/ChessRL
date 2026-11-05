"""The d_model-512 INT8 kernel (cpp/net/encoder_fused512.cu) on a GPU.

Against the INT8 specification (src/export/int8.py's int8_reference: exact integer sums, the attention output
rounded per head, the SwiGLU output per 128 columns) and the FP32 network on random planes, with both value-head
mixes; its packed input path against the dense planes of the CPU encoder, which must give identical outputs. The
kernel is not bit-exact with the specification (an FP16 residual stream and FP16 policy q / k, FP16 sums in P · V,
the QK-norm applied as row scales of the scores, approximate exp2 and SiLU): its FP16 values flip some of the
specification's INT8 roundings, so it is held to the specification's own accuracy against FP32.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

_inference_cpp = pytest.importorskip("_inference_cpp")

from src.export.heads import ExportWrapper  # noqa: E402
from src.export.int8 import int8_reference  # noqa: E402
from src.export.to_cnet import entries_for, write_cnet  # noqa: E402
from src.models.encoder import EncoderModel  # noqa: E402
from tests.test_fused_network import CASES, STARTPOS, random_games  # noqa: E402

pytestmark = pytest.mark.gpu


def model_512(layers: int = 2) -> EncoderModel:
    torch.manual_seed(0)
    model = EncoderModel(d_model=512, layers=layers, heads=8, d_ff=768, int8=True).eval()
    with torch.no_grad():  # the trained-only parameters start at zero; make them count
        model.pair_bias.normal_(0, 1.0)
        model.promo.normal_(0, 0.5)
        for blk in model.blocks:
            blk.x0_gain.fill_(0.3)
    return model


def planes(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = (rng.random((n, 8, 8, 111)) < 0.1).astype(np.float32)
    p[..., 110] = 0.37
    return p


def fp32_copy(model: EncoderModel) -> EncoderModel:
    """The same weights without INT8 rounding (an int8 model's own forward rounds as the kernel)."""
    m = copy.deepcopy(model)
    m.int8 = False
    for b in m.blocks:
        b.int8 = False
    return m


def rms(a: np.ndarray) -> float:
    return float(np.sqrt((np.asarray(a, np.float64) ** 2).mean()))


@pytest.fixture(scope="module")
def net(tmp_path_factory):
    model = model_512()
    path = tmp_path_factory.mktemp("cnet") / "f512.cnet"
    write_cnet(path, entries_for(model, "search", int8=True))
    return model, str(path)


def test_as_close_to_fp32_as_the_int8_specification(net):
    """The kernel and the specification are two roundings of the same INT8 network (the kernel's FP16 intermediates
    flip some of the specification's INT8 roundings): each as far from the FP32 network as the other."""
    model, path = net
    x = planes(32, 0)
    out = _inference_cpp.fused_forward_dense(path, x)
    with torch.no_grad():
        spec = [t.numpy() for t in ExportWrapper(int8_reference(model), "search")(torch.from_numpy(x))]
        fp32 = [t.numpy() for t in ExportWrapper(fp32_copy(model), "search")(torch.from_numpy(x))]
    assert np.isfinite(out["policy"]).all() and np.isfinite(out["value"]).all()
    p, s, f = (a.astype(np.float64) for a in (out["policy"], spec[0], fp32[0]))
    assert rms(p - f) < 1.2 * rms(s - f), (rms(p - f), rms(s - f))
    assert rms(p - s) < 0.05 * float(s.std())
    assert (p.argmax(1) == f.argmax(1)).mean() >= (s.argmax(1) == f.argmax(1)).mean() - 0.07
    assert np.abs(out["value"] - fp32[1]).max() < 2 * np.abs(spec[1] - fp32[1]).max() + 1e-4
    assert np.abs(out["mlh"] - fp32[2]).max() < 0.05
    assert np.abs(out["mlh_q"] - fp32[3]).max() < 0.05


def test_blend_heads(tmp_path):
    """search_heads blend: the WDL mixes both value heads' softmaxes (wdl_w), mlh_q repeats mlh."""
    model = model_512(layers=1)
    path = tmp_path / "blend.cnet"
    write_cnet(path, entries_for(model, "blend", q_weight=0.3, int8=True))
    x = planes(4, 1)
    out = _inference_cpp.fused_forward_dense(str(path), x)
    with torch.no_grad():
        spec = [t.numpy() for t in ExportWrapper(int8_reference(model), "blend", 0.3)(torch.from_numpy(x))]
    assert np.abs(out["value"] - spec[1]).max() < 2e-3
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


def test_batches_are_independent(net):
    """One position's record does not depend on its neighbours (per-block scratch, barriers between phases)."""
    _, path = net
    x = planes(64, 2)
    whole = _inference_cpp.fused_forward_dense(path, x)
    alone = _inference_cpp.fused_forward_dense(path, x[5:6])
    for key in whole:
        assert np.array_equal(whole[key][5:6], alone[key]), key
