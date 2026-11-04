"""INT8 projections (src/export/int8.py, to_cnet --int8): the fragment layout read back from the PTX definition of
mma.sync.m16n8k32's B operand, the file's weights against the reference's, and the reference against the FP32
network. CPU only; scripts/net_parity.py --int8 checks the GPU kernel against the reference.
"""

from __future__ import annotations

import numpy as np
import torch

from src.export.heads import ExportWrapper
from src.export.int8 import int8_reference, quantize_rows
from src.export.to_cnet import entries_for, export, pack_b8, write_cnet
from src.models.encoder import EncoderModel
from tests.test_cnet_export import read_cnet


def unpack_b8(flat: np.ndarray, n: int, k: int) -> np.ndarray:
    """B fragments of m16n8k32 (.s8, .col): b0 = B[4t..4t+3][g], b1 = B[16+4t..16+4t+3][g]; one uint4 per lane =
    (b0, b1) of k-step 2p, then of k-step 2p+1. B[k][n] = W[n][k]."""
    frag = np.ascontiguousarray(flat).view(np.int8).reshape(n // 8, k // 64, 32, 16)
    w = np.zeros((n, k), np.int8)
    for j in range(n // 8):
        for p in range(k // 64):
            for lane in range(32):
                g, t = lane // 4, lane % 4
                for s in range(2):
                    for b in range(2):
                        for e in range(4):
                            w[8 * j + g, 64 * p + 32 * s + 16 * b + 4 * t + e] = frag[j, p, lane, 8 * s + 4 * b + e]
    return w


def test_fragments_round_trip():
    rng = np.random.default_rng(0)
    for n, k in ((384, 128), (128, 256), (8, 64)):
        q = rng.integers(-127, 128, (n, k)).astype(np.int8)
        packed = pack_b8(q)
        assert packed.dtype == np.int32 and packed.size * 4 == n * k
        assert np.array_equal(unpack_b8(packed, n, k), q)


def test_quantize_rows():
    rng = np.random.default_rng(1)
    w = rng.normal(0, 0.05, (64, 128)).astype(np.float32)
    w[3] = 0.0  # an all-zero row keeps a finite scale
    q, s = quantize_rows(w)
    assert q.dtype == np.int8 and np.abs(q).max() == 127 and np.isfinite(s).all()
    assert (np.abs(q.astype(np.float32) * s[:, None] - w) <= s[:, None] / 2 + 1e-9).all()


def _model():
    torch.manual_seed(0)
    model = EncoderModel().eval()
    with torch.no_grad():
        model.pair_bias.normal_(0, 1.0)
        model.promo.normal_(0, 0.5)
        for blk in model.blocks:
            blk.x0_gain.fill_(0.3)
    return model


def test_file_weights_are_the_reference_weights(tmp_path):
    model = _model()
    path = tmp_path / "net.cnet"
    write_cnet(path, entries_for(model, "search", int8=True))
    f = read_cnet(path)
    assert int(f["int8"][0]) == 1
    ref = int8_reference(model)
    d, d_ff = 128, 256

    def same(name, mod, n, k, cols=None):
        q = unpack_b8(f[name], n, k).astype(np.float32)
        if cols is not None:
            q = q[:, :cols]
        assert np.array_equal(q, mod.qw.numpy()), name
        assert np.array_equal(f[name + ".s"], mod.sw.numpy()), name

    same("embed.w", ref.embed, d, 128, cols=ref.embed.qw.shape[1])
    for i, b in enumerate(ref.blocks):
        same(f"l{i}.qkv", b.qkv, 3 * d, d)
        same(f"l{i}.out", b.out, d, d)
        same(f"l{i}.up", b.up, 2 * d_ff, d)
        same(f"l{i}.down", b.down, d, d_ff)
    # The policy's q and k projections share one matrix (q pre-scaled by 1/sqrt(d): the same integers per row).
    q = unpack_b8(f["pol.w"], 2 * d, d).astype(np.float32)
    assert np.array_equal(q[:d], ref.pol_q.qw.numpy()) and np.array_equal(q[d:], ref.pol_k.qw.numpy())


def test_export_follows_model_int8(tmp_path):
    """export() writes INT8 projections when the checkpoint's run config says model.int8 (build_net.sh, rerate)."""
    model = _model()
    for flag in (False, True):
        ckpt = {"model": model.state_dict(), "model_spec": model.spec(),
                "config": {"search_heads": "outcome", "model": {"int8": flag}}}
        torch.save(ckpt, tmp_path / "c.pt")
        assert export(tmp_path / "c.pt", tmp_path / "c.cnet").endswith(" int8") == flag
        f = read_cnet(tmp_path / "c.cnet")
        assert ("int8" in f) == flag and ("l0.qkv.s" in f) == flag
    assert not export(tmp_path / "c.pt", tmp_path / "d.cnet", int8=False).endswith(" int8")


def test_reference_close_to_fp32():
    model = _model()
    rng = np.random.default_rng(0)
    planes = (rng.random((4, 64, 111)) < 0.1).astype(np.float32)
    planes[:, :, 110] = 0.37
    planes[:, :, 104] = 1.0
    x = torch.from_numpy(planes).reshape(4, 8, 8, 111)
    with torch.no_grad():
        ref = ExportWrapper(model, "search", 0.5)(x)
        q = ExportWrapper(int8_reference(model), "search", 0.5)(x)
    lp, lq = torch.log_softmax(ref[0], -1), torch.log_softmax(q[0], -1)
    kl = (lp.exp() * (lp - lq)).sum(-1)
    assert kl.max() < 1e-2, kl
    assert (ref[1] - q[1]).abs().max() < 2e-2
