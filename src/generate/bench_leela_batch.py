import argparse
import time
import torch

from src.models.leela.model import BT4 as LeelaBT4


def build_move_lists(batch_size: int) -> list[list[str]]:
    # Simple legal opening sequence from start position
    base = [
        "e2e4",
        "e7e5",
        "g1f3",
        "b8c6",
        "f1b5",
        "a7a6",
    ]
    return [base[:] for _ in range(batch_size)]


def main():
    p = argparse.ArgumentParser(description="Benchmark Leela batch inference on move lists")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--temp", type=float, default=0.0)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--fp16", action="store_true")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    model = LeelaBT4().to(device)
    if args.fp16 and device.type == 'cuda':
        model = model.half()
    if args.ckpt:
        sd = torch.load(args.ckpt, map_location=device)
        state = sd.get("state_dict", sd)
        model.load_state_dict(state, strict=False)
    model.eval()

    move_lists = build_move_lists(args.batch_size)

    # Warmup
    for _ in range(args.warmup):
        _ = model.batch_get_moves_from_move_lists(move_lists, T=args.temp, device=str(device), use_fp16=args.fp16)

    times_ms: list[float] = []
    for _ in range(args.runs):
        t0 = time.time()
        _ = model.batch_get_moves_from_move_lists(move_lists, T=args.temp, device=str(device), use_fp16=args.fp16)
        dt_ms = (time.time() - t0) * 1000.0
        times_ms.append(dt_ms)

    avg = sum(times_ms) / len(times_ms)
    print(f"Leela batch_get_moves_from_move_lists | batch={args.batch_size} runs={args.runs} avg_ms={avg:.2f}")
    print("runs_ms="," ".join(f"{t:.2f}" for t in times_ms))


if __name__ == "__main__":
    main()


