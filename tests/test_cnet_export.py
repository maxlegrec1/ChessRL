"""The .cnet file (src/export/to_cnet.py) read the way cpp/net/encoder_kernel.cu reads it.

Unpacks every fragment layout from the PTX definitions of the mma.sync operands, replays
the kernel's arithmetic in NumPy (folded scales, exp2 softmax, the vocab codes) and
compares with the PyTorch network. CPU only; the GPU kernel itself is checked against
PyTorch by scripts/net_parity.py.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest
import torch

from src.export.to_cnet import HEADS, entries_for, write_cnet
from src.export.to_onnx import ExportWrapper
from src.models.encoder import EncoderModel

DT = {0: np.float16, 1: np.float32, 2: np.uint16, 3: np.int32}


def read_cnet(path) -> dict[str, np.ndarray]:
    data = path.read_bytes()
    assert data[:8] == b"CHESSNET"
    version, count = struct.unpack_from("<II", data, 8)
    assert version == 1
    out = {}
    for i in range(count):
        name, dtype, _, off, nbytes = struct.unpack_from("<32sIIQQ", data, 16 + 56 * i)
        assert off % 256 == 0
        out[name.rstrip(b"\0").decode()] = np.frombuffer(data, DT[dtype], nbytes // np.dtype(DT[dtype]).itemsize, off)
    return out


def unpack_b(flat: np.ndarray, n: int, k: int) -> np.ndarray:
    """B fragments of m16n8k16 (.col): b0 = B[2t, 2t+1][g], b1 = B[2t+8, 2t+9][g]; one uint4
    per lane = (b0, b1) of k-step 2p, then of k-step 2p+1. B[k][n] = W[n][k]."""
    frag = flat.reshape(n // 8, k // 32, 32, 8)
    w = np.zeros((n, k), np.float32)
    for j in range(n // 8):
        for p in range(k // 32):
            for lane in range(32):
                g, t = lane // 4, lane % 4
                for s in range(2):
                    for b in range(2):
                        for e in range(2):
                            w[8 * j + g, 32 * p + 16 * s + 8 * b + 2 * t + e] = frag[j, p, lane, 4 * s + 2 * b + e]
    return w


def unpack_bias(flat: np.ndarray, heads: int) -> np.ndarray:
    """Accumulator fragments: c0, c1 = row g, cols 2t, 2t+1; c2, c3 = row g+8, same cols."""
    frag = flat.reshape(heads, 4, 4, 32, 8)
    bias = np.zeros((heads, 64, 64), np.float32)
    for h in range(heads):
        for m in range(4):
            for c in range(4):
                for lane in range(32):
                    g, t = lane // 4, lane % 4
                    for v in range(8):
                        j, r = 2 * c + v // 4, v % 4
                        bias[h, 16 * m + g + 8 * (r // 2), 8 * j + 2 * t + r % 2] = frag[h, m, c, lane, v]
    return bias


def rms(x, w, eps=1e-6):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w


def silu(x):
    return x / (1 + np.exp(-x))


def kernel_forward(f: dict[str, np.ndarray], planes: np.ndarray):
    """The kernel's arithmetic for one batch [B, 64, planes], in FP32 from the packed file."""
    d, layers, heads, d_ff, n_planes, n_wdl, mlhq = (int(v) for v in f["config"])
    hd = d // heads
    a = np.zeros((planes.shape[0], 64, 128), np.float32)
    a[:, :, :n_planes] = planes
    x0 = a @ unpack_b(f["embed.w"], d, 128).T + f["embed.b"] + f["pos"].reshape(64, d)
    bias = unpack_bias(f["pair_bias"], heads)
    x = x0
    for i in range(layers):
        x = x + float(f[f"l{i}.gain"][0]) * x0
        qkv = rms(x, f[f"l{i}.norm1"]) @ unpack_b(f[f"l{i}.qkv"], 3 * d, d).T
        q, k, v = (qkv[..., j * d : (j + 1) * d].reshape(-1, 64, heads, hd) for j in range(3))
        q, k = rms(q, f[f"l{i}.qnorm"]), rms(k, f[f"l{i}.knorm"])
        s = np.einsum("bqhd,bkhd->bhqk", q, k) + bias
        p = np.exp2(s - s.max(-1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        o = np.einsum("bhqk,bkhd->bqhd", p, v).reshape(-1, 64, d)
        x = x + o @ unpack_b(f[f"l{i}.out"], d, d).T
        u = rms(x, f[f"l{i}.norm2"]) @ unpack_b(f[f"l{i}.up"], 2 * d_ff, d).T
        x = x + (silu(u[..., :d_ff]) * u[..., d_ff:]) @ unpack_b(f[f"l{i}.down"], d, d_ff).T
    h = rms(x, f["norm"])
    qk = h @ unpack_b(f["pol.w"], 2 * d, d).T + f["pol.b"]
    q, k = qk[..., :d], qk[..., d:]
    ft = q @ k.transpose(0, 2, 1)
    pt = f["promo"].astype(np.float32).reshape(4, d) @ k.transpose(0, 2, 1)
    code = f["vocab"].astype(np.int64)
    frm, to, promo, special = code & 63, (code >> 6) & 63, (code >> 12) & 3, code >> 15
    logits = ft[:, frm, to] + pt[:, promo, to]
    logits = np.where(special == 1, f["special"][0], logits)

    pooled = h.mean(1)
    n_mlp = n_wdl + 1 + mlhq
    w1 = f["mlp1.w"].astype(np.float32).reshape(n_mlp, d, d)
    hid = silu(np.einsum("bi,mio->bmo", pooled, w1) + f["mlp1.b"].reshape(n_mlp, d))
    w2 = f["mlp2.w"].reshape(-1, d)
    mlp_of = [o // 3 for o in range(3 * n_wdl)] + [n_wdl + j for j in range(1 + mlhq)]
    outs = np.stack([hid[:, m] @ w2[o] for o, m in enumerate(mlp_of)], 1) + f["mlp2.b"]
    wdl = 0
    for j in range(n_wdl):
        e = np.exp(outs[:, 3 * j : 3 * j + 3] - outs[:, 3 * j : 3 * j + 3].max(1, keepdims=True))
        wdl = wdl + e / e.sum(1, keepdims=True) / n_wdl
    mlh = outs[:, 3 * n_wdl]
    return logits, wdl, mlh, outs[:, 3 * n_wdl + 1] if mlhq else mlh


@pytest.mark.parametrize("heads", sorted(HEADS))
def test_packed_file_matches_pytorch(tmp_path, heads):
    torch.manual_seed(0)
    model = EncoderModel().eval()
    with torch.no_grad():  # the trained-only parameters start at zero; make them count
        model.pair_bias.normal_(0, 1.0)
        model.promo.normal_(0, 0.5)
        for blk in model.blocks:
            blk.x0_gain.fill_(0.3)
            blk.q_norm.weight.uniform_(0.5, 1.5)
            blk.k_norm.weight.uniform_(0.5, 1.5)
    path = tmp_path / "net.cnet"
    write_cnet(path, entries_for(model, heads))
    f = read_cnet(path)

    rng = np.random.default_rng(0)
    planes = (rng.random((3, 64, 111)) < 0.1).astype(np.float32)
    planes[:, :, 110] = 0.37  # the halfmove plane is fractional
    got = kernel_forward(f, planes)
    with torch.no_grad():
        ref = [t.numpy() for t in ExportWrapper(model, heads)(torch.from_numpy(planes).reshape(3, 8, 8, 111))]
    # FP16 weights against FP32: logits within a few hundredths, probabilities closer.
    assert np.abs(got[0] - ref[0]).max() < 0.05
    assert np.abs(got[1] - ref[1]).max() < 5e-3
    assert np.abs(got[2] - ref[2]).max() < 0.02
    assert np.abs(got[3] - ref[3]).max() < 0.02
