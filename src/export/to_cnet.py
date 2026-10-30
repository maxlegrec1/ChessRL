"""Export an encoder checkpoint to the fused CUDA network format (.cnet).

The file holds exactly what cpp/net/encoder_kernel.cu reads, so the C++ side only copies
it to the GPU: weight matrices in FP16, pre-packed in the order mma.sync wants its B
fragments, FP32 vectors, the pair bias in attention-score fragment order, and the policy
vocab as one 16-bit code per entry. Two scales are folded in: the attention one
(1/sqrt(head_dim), and log2(e) because the kernel's softmax uses exp2) and the policy
one (1/sqrt(d_model), into pol_q and the promotion vectors).

Layout: b"CHESSNET", u32 version, u32 entry count, then per entry a 32-byte name, u32
dtype (0 f16, 1 f32, 2 u16, 3 i32), u32 zero, u64 offset, u64 bytes; each entry's data
starts on a 256-byte boundary.

    uv run python -m src.export.to_cnet --ckpt_path ckpt.pt --out net.cnet
"""

from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path

import numpy as np
import torch

from src.models.encoder import _vocab_tables
from src.models.factory import model_from_checkpoint

MAGIC = b"CHESSNET"
VERSION = 1
DTYPES = {np.float16: 0, np.float32: 1, np.uint16: 2, np.int32: 3}
LOG2E = 1.0 / math.log(2.0)
# ExportWrapper's modes: the MLPs whose softmax is averaged into "value", and whether
# "mlh_q" comes from the mlh_q head (otherwise it repeats mlh).
HEADS = {
    "search": (["value_q"], True),
    "outcome": (["value"], False),
    "q_value": (["value_q"], False),
    "blend": (["value", "value_q"], False),
}


def pack_b(w: np.ndarray) -> np.ndarray:
    """A Linear weight [N, K] as m16n8k16 B fragments: [N/8 tiles][K/32][32 lanes][8 halfs].

    Lane l (g = l // 4, t = l % 4) of n-tile j holds column n = 8j + g at k offsets
    2t, 2t+1, 2t+8, 2t+9 (one k-step's b0, b1) and the same plus 16 (the next k-step),
    so one 16-byte load per lane fetches two k-steps."""
    n, k = w.shape
    assert n % 8 == 0 and k % 32 == 0, w.shape
    j = np.arange(n // 8)[:, None, None, None]
    p = np.arange(k // 32)[None, :, None, None]
    lane = np.arange(32)[None, None, :, None]
    e = np.arange(8)[None, None, None, :]
    g, t = lane // 4, lane % 4
    kk = 2 * t + (e & 1) + 8 * ((e >> 1) & 1) + 16 * (e >> 2)
    return np.ascontiguousarray(w[8 * j + g, 32 * p + kk], dtype=np.float16)


def pack_pair_bias(bias: np.ndarray) -> np.ndarray:
    """[heads, 64 queries, 64 keys] as score fragments: [head][query tile 4][chunk 4][lane][8].

    Chunk c covers key tiles 2c and 2c+1; element v is key tile 2c + v // 4, accumulator
    register r = v % 4 (rows g / g+8 for r < 2 / r >= 2, key column 2t + r % 2)."""
    h = np.arange(bias.shape[0])[:, None, None, None, None]
    m = np.arange(4)[None, :, None, None, None]
    c = np.arange(4)[None, None, :, None, None]
    lane = np.arange(32)[None, None, None, :, None]
    v = np.arange(8)[None, None, None, None, :]
    g, t = lane // 4, lane % 4
    r = v % 4
    row = 16 * m + g + 8 * (r // 2)
    col = 8 * (2 * c + v // 4) + 2 * t + r % 2
    return np.ascontiguousarray(bias[h, row, col], dtype=np.float16)


def vocab_codes() -> np.ndarray:
    """Per vocab entry: from token | to token << 6 | promotion << 12 | special << 15."""
    frm, to, promo = (x.numpy() for x in _vocab_tables())
    special = promo < 0
    codes = frm | (to << 6) | (np.maximum(promo, 0) << 12) | (special.astype(np.int64) << 15)
    return codes.astype(np.uint16)


def entries_for(model, search_heads: str) -> list[tuple[str, np.ndarray]]:
    cfg = model.cfg
    d, layers, heads, d_ff = cfg["d_model"], cfg["layers"], cfg["heads"], cfg["d_ff"]
    hd = d // heads
    sd = {k: v.detach().float().cpu().numpy() for k, v in model.state_dict().items()}
    wdl_mlps, mlhq_from_q = HEADS[search_heads]
    mlps = [*wdl_mlps, "mlh"] + (["mlh_q"] if mlhq_from_q else [])
    planes = sd["embed.weight"].shape[1]
    embed = np.zeros((d, 128), np.float32)  # input planes padded to the 128-wide k loop
    embed[:, :planes] = sd["embed.weight"]
    pol_scale = 1.0 / math.sqrt(d)

    out: list[tuple[str, np.ndarray]] = [
        (
            "config",
            np.array(
                [d, layers, heads, d_ff, planes, len(wdl_mlps), int(mlhq_from_q)], np.int32
            ),
        ),
        ("embed.w", pack_b(embed)),
        ("embed.b", sd["embed.bias"]),
        ("pos", sd["pos"]),
        ("pair_bias", pack_pair_bias(sd["pair_bias"] * LOG2E)),
    ]
    for i in range(layers):
        b = f"blocks.{i}."
        out += [
            (f"l{i}.norm1", sd[b + "norm1.weight"]),
            (f"l{i}.qkv", pack_b(sd[b + "qkv.weight"])),
            (f"l{i}.qnorm", sd[b + "q_norm.weight"] * (LOG2E / math.sqrt(hd))),
            (f"l{i}.knorm", sd[b + "k_norm.weight"]),
            (f"l{i}.out", pack_b(sd[b + "out.weight"])),
            (f"l{i}.norm2", sd[b + "norm2.weight"]),
            (f"l{i}.up", pack_b(sd[b + "up.weight"])),
            (f"l{i}.down", pack_b(sd[b + "down.weight"])),
            (f"l{i}.gain", sd[b + "x0_gain"].reshape(1)),
        ]
    pol_w = np.concatenate([sd["pol_q.weight"] * pol_scale, sd["pol_k.weight"]])
    pol_b = np.concatenate([sd["pol_q.bias"] * pol_scale, sd["pol_k.bias"]])
    out += [
        ("norm", sd["norm.weight"]),
        ("pol.w", pack_b(pol_w)),
        ("pol.b", pol_b),
        ("promo", (sd["promo"] * pol_scale).astype(np.float16)),
        ("special", sd["special_logit"].reshape(1)),
        ("vocab", vocab_codes()),
        # First layers transposed to [mlp][in][out] so the kernel's per-unit loop reads
        # consecutive units per thread; second layers stacked in output order.
        ("mlp1.w", np.stack([sd[f"{m}.0.weight"].T for m in mlps]).astype(np.float16)),
        ("mlp1.b", np.concatenate([sd[f"{m}.0.bias"] for m in mlps])),
        ("mlp2.w", np.concatenate([sd[f"{m}.2.weight"] for m in mlps])),
        ("mlp2.b", np.concatenate([sd[f"{m}.2.bias"] for m in mlps])),
    ]
    return out


def write_cnet(path: Path, entries: list[tuple[str, np.ndarray]]) -> None:
    head = 16 + 56 * len(entries)
    offset = -(-head // 256) * 256
    table, blobs = [], []
    for name, arr in entries:
        arr = np.ascontiguousarray(arr)
        if arr.dtype == np.float64:
            arr = arr.astype(np.float32)
        code = DTYPES[arr.dtype.type]
        name_b = name.encode()
        assert len(name_b) < 32, name
        table.append(struct.pack("<32sIIQQ", name_b, code, 0, offset, arr.nbytes))
        blobs.append((offset, arr.tobytes()))
        offset = -(-(offset + arr.nbytes) // 256) * 256
    data = bytearray(offset)
    data[:16] = MAGIC + struct.pack("<II", VERSION, len(entries))
    data[16:head] = b"".join(table)
    for off, blob in blobs:
        data[off : off + len(blob)] = blob
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(bytes(data))
    tmp.replace(path)  # self-play may be watching for the file


def export(ckpt_path: Path, out_path: Path, search_heads: str | None = None) -> str:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not (isinstance(ckpt, dict) and "model" in ckpt):
        ckpt = {"model": ckpt}
    model = model_from_checkpoint(ckpt, "cpu").eval()
    if not hasattr(model, "cfg"):
        raise SystemExit("the fused network supports the encoder architecture only")
    heads = search_heads or (ckpt.get("config") or {}).get("search_heads", "search")
    write_cnet(out_path, entries_for(model, heads))
    return heads


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--ckpt_path", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--search_heads", default=None, choices=sorted(HEADS))
    a = p.parse_args()
    heads = export(a.ckpt_path, a.out, a.search_heads)
    print(f"[to_cnet] {a.ckpt_path} -> {a.out} ({a.out.stat().st_size} bytes, heads={heads})")


if __name__ == "__main__":
    main()
