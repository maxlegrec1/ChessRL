"""Export an encoder checkpoint to the fused CUDA network format (.cnet).

The file holds exactly what cpp/net/encoder_kernel.cu reads, so the C++ side only copies
it to the GPU: weight matrices in FP16, pre-packed in the order mma.sync wants its B
fragments, FP32 vectors, the pair bias in attention-score fragment order, and the policy
vocab as one 16-bit code per entry, and the weights of the value MLPs in the WDL the search
reads (wdl_w; search_heads blend). Two scales are folded in: the attention one
(1/sqrt(head_dim), and log2(e) because the kernel's softmax uses exp2) and the policy
one (1/sqrt(d_model), into pol_q and the promotion vectors).

With --int8 the projections (embedding, qkv, out, up, down, policy q/k) are INT8 with one FP32 scale per output
row (src/export/int8.py): <name> holds pack_b8 fragments as int32 words, <name>.s the scales, and an "int8" entry
tells the kernel to run its INT8 path.

Layout: b"CHESSNET", u32 version, u32 entry count, then per entry a 32-byte name, u32
dtype (0 f16, 1 f32, 2 u16, 3 i32), u32 zero, u64 offset, u64 bytes; each entry's data
starts on a 256-byte boundary.

    uv run python -m src.export.to_cnet --ckpt_path ckpt.pt --out net.cnet [--int8]
"""

from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path

import numpy as np
import torch

from src.export.heads import search_q_weight
from src.export.int8 import quantize_rows
from src.models.encoder import _vocab_tables
from src.models.factory import model_from_checkpoint

MAGIC = b"CHESSNET"
VERSION = 2  # 2: the canonical 1,858-move vocabulary (1: 1,929 absolute moves and 5 special tokens)
DTYPES = {np.float16: 0, np.float32: 1, np.uint16: 2, np.int32: 3}
LOG2E = 1.0 / math.log(2.0)
# src/export/heads.py's ExportWrapper modes: the MLPs whose softmaxes are mixed into "value" (weights wdl_w),
# and whether "mlh_q" comes from the mlh_q head (otherwise it repeats mlh).
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


def pack_b8(q: np.ndarray) -> np.ndarray:
    """An INT8 weight [N, K] as m16n8k32 B fragments: [N/8 tiles][K/64][32 lanes][16 bytes], as int32 words.

    Lane l (g = l // 4, t = l % 4) of n-tile j holds column n = 8j + g at k offsets 4t..4t+3 and 16+4t..16+4t+3
    (one k32 step's b0, b1) and the same plus 32 (the next step), so one 16-byte load per lane fetches 64 k."""
    n, k = q.shape
    assert n % 8 == 0 and k % 64 == 0 and q.dtype == np.int8, (q.shape, q.dtype)
    j = np.arange(n // 8)[:, None, None, None]
    p = np.arange(k // 64)[None, :, None, None]
    lane = np.arange(32)[None, None, :, None]
    e = np.arange(16)[None, None, None, :]
    g, t = lane // 4, lane % 4
    kk = 4 * t + (e & 3) + 16 * ((e >> 2) & 1) + 32 * (e >> 3)
    return np.ascontiguousarray(q[8 * j + g, 64 * p + kk]).view(np.int32)


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
    """Per vocab entry: from token | to token << 6 | promotion << 12."""
    frm, to, promo = (x.numpy() for x in _vocab_tables())
    return (frm | (to << 6) | (promo << 12)).astype(np.uint16)


# The sizes cpp/net/encoder_kernel.cu is compiled for (any depth up to MAX_LAYERS).
KERNEL_DIMS = {"d_model": 128, "heads": 8, "d_ff": 256}
MAX_LAYERS = 16


FP16_IN_INT8 = ("embed", "out", "down")  # the projections an int8 network may keep in FP16


def entries_for(
    model, search_heads: str, q_weight: float = 0.5, int8: bool = False, fp16: tuple[str, ...] = ()
) -> list[tuple[str, np.ndarray]]:
    """The .cnet entries; q_weight is value_q's weight in the WDL under search_heads blend; int8: INT8
    projections, but those named in fp16 (some of FP16_IN_INT8)."""
    if not set(fp16) <= set(FP16_IN_INT8):
        raise ValueError(f"an int8 network keeps only {FP16_IN_INT8} in FP16, not {fp16}")
    cfg = model.cfg
    d, layers, heads, d_ff = cfg["d_model"], cfg["layers"], cfg["heads"], cfg["d_ff"]
    if any(cfg[k] != v for k, v in KERNEL_DIMS.items()) or not 1 <= layers <= MAX_LAYERS:
        raise ValueError(
            f"the fused kernel runs d_model 128, 8 heads, d_ff 256 and 1 to {MAX_LAYERS} layers; "
            f"this network has {cfg}"
        )
    hd = d // heads
    sd = {k: v.detach().float().cpu().numpy() for k, v in model.state_dict().items()}
    wdl_mlps, mlhq_from_q = HEADS[search_heads]
    wdl_w = [1.0 - q_weight, q_weight] if search_heads == "blend" else [1.0]
    mlps = [*wdl_mlps, "mlh"] + (["mlh_q"] if mlhq_from_q else [])
    planes = sd["embed.weight"].shape[1]
    embed = np.zeros((d, 128), np.float32)  # input planes padded to the 128-wide k loop
    embed[:, :planes] = sd["embed.weight"]
    pol_scale = 1.0 / math.sqrt(d)

    def mat(name: str, w: np.ndarray, fp16: bool = False) -> list[tuple[str, np.ndarray]]:
        """A projection: FP16 B fragments, or INT8 ones and their per-row scales."""
        if not int8 or fp16:
            return [(name, pack_b(w))]
        q, scale = quantize_rows(w)
        return [(name, pack_b8(q)), (name + ".s", scale)]

    out: list[tuple[str, np.ndarray]] = [
        (
            "config",
            np.array([d, layers, heads, d_ff, planes, len(wdl_mlps), int(mlhq_from_q)], np.int32),
        ),
        *mat("embed.w", embed, fp16="embed" in fp16),
        ("embed.b", sd["embed.bias"]),
        ("pos", sd["pos"]),
        ("pair_bias", pack_pair_bias(sd["pair_bias"] * LOG2E)),
    ]
    for i in range(layers):
        b = f"blocks.{i}."
        out += [
            (f"l{i}.norm1", sd[b + "norm1.weight"]),
            *mat(f"l{i}.qkv", sd[b + "qkv.weight"]),
            (f"l{i}.qnorm", sd[b + "q_norm.weight"] * (LOG2E / math.sqrt(hd))),
            (f"l{i}.knorm", sd[b + "k_norm.weight"]),
            *mat(f"l{i}.out", sd[b + "out.weight"], fp16="out" in fp16),
            (f"l{i}.norm2", sd[b + "norm2.weight"]),
            *mat(f"l{i}.up", sd[b + "up.weight"]),
            *mat(f"l{i}.down", sd[b + "down.weight"], fp16="down" in fp16),
            (f"l{i}.gain", sd[b + "x0_gain"].reshape(1)),
        ]
    pol_w = np.concatenate([sd["pol_q.weight"] * pol_scale, sd["pol_k.weight"]])
    pol_b = np.concatenate([sd["pol_q.bias"] * pol_scale, sd["pol_k.bias"]])
    out += [
        ("norm", sd["norm.weight"]),
        *mat("pol.w", pol_w),
        ("pol.b", pol_b),
        ("promo", (sd["promo"] * pol_scale).astype(np.float16)),
        ("vocab", vocab_codes()),
        ("wdl_w", np.array(wdl_w, np.float32)),
        # First layers transposed to [mlp][in][out] so the kernel's per-unit loop reads
        # consecutive units per thread; second layers stacked in output order.
        ("mlp1.w", np.stack([sd[f"{m}.0.weight"].T for m in mlps]).astype(np.float16)),
        ("mlp1.b", np.concatenate([sd[f"{m}.0.bias"] for m in mlps])),
        ("mlp2.w", np.concatenate([sd[f"{m}.2.weight"] for m in mlps])),
        ("mlp2.b", np.concatenate([sd[f"{m}.2.bias"] for m in mlps])),
    ]
    # Optional: input plane 104 is the real side to move (model.stm_plane colour). Without the entry the kernel
    # sets it to 1, as every network before this setting was trained.
    if cfg.get("stm_plane", "constant") == "colour":
        out.append(("stm_colour", np.array([1], np.int32)))
    if int8:
        out.append(("int8", np.array([1], np.int32)))
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


def export(
    ckpt_path: Path,
    out_path: Path,
    search_heads: str | None = None,
    int8: bool | None = None,
    fp16: tuple[str, ...] = (),
) -> str:
    """Write the .cnet; returns the heads it reads ("blend 0.25": value_q's weight). int8 None: INT8 when the
    network was trained so (its model_spec) or the run asks for it (the config's model.int8)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not (isinstance(ckpt, dict) and "model" in ckpt):
        ckpt = {"model": ckpt}
    model = model_from_checkpoint(ckpt, "cpu").eval()
    if not hasattr(model, "cfg"):
        raise SystemExit("the fused network supports the encoder architecture only")
    config = ckpt.get("config") or {}
    heads = search_heads or config.get("search_heads", "search")
    if int8 is None:  # the network's own setting (INT8 training), else the run's (INT8 export of an FP16 network)
        int8 = bool(model.cfg.get("int8") or (config.get("model") or {}).get("int8", False))
    q_weight = search_q_weight(ckpt)
    write_cnet(out_path, entries_for(model, heads, q_weight, int8, tuple(fp16)))
    kind = " int8" if int8 else ""
    return (f"{heads} {q_weight:.3f}" if heads == "blend" else heads) + kind


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--ckpt_path", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--search_heads", default=None, choices=sorted(HEADS))
    p.add_argument("--int8", action="store_true", default=None,
                   help="INT8 projections (src/export/int8.py); default: the checkpoint's model.int8")
    p.add_argument("--fp16", default="", help=f"with --int8: projections kept FP16, comma-separated, of {FP16_IN_INT8}")
    a = p.parse_args()
    heads = export(a.ckpt_path, a.out, a.search_heads, a.int8, tuple(x for x in a.fp16.split(",") if x))
    print(f"[to_cnet] {a.ckpt_path} -> {a.out} ({a.out.stat().st_size} bytes, heads={heads})")


if __name__ == "__main__":
    main()
