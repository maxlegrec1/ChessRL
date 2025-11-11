import argparse
import os
import torch

from .arena import SelfPlayConfig, SelfPlayArena
from .ui import run as run_ui
from .openings import hf_opening_fen_generator

# Models
from src.models.small.inference_cpp import CppPolicy
from src.models.leela.model import BT4 as LeelaBT4


class _CppPolicyAdapter:
    """
    Adapter to provide the same interface expected by SelfPlayArena for the C++ policy.
    """

    def __init__(self, engine_path: str = "model.trt"):
        self.cpp_policy = CppPolicy(engine_path=engine_path)

    def batch_get_moves_from_fens(
        self,
        fens,
        T: float = 0.0,
        device: str = None,
        force_legal: bool = True,
        return_probs: bool = False,
        use_fp16: bool = False,
        boards=None,
        topk_prefilter: int = 0,
    ):
        # The C++ policy always returns legal moves, so ignore force_legal/topk options.
        original_length = len(fens)
        if len(fens) != 1024:
            fens = fens + [fens[0]] * (1024 - len(fens))
        moves = self.cpp_policy(fens, temperature=T)
        moves = moves[:original_length]
        if return_probs:
            raise NotImplementedError("CppPolicy does not currently support returning probabilities")
        return moves

    def eval(self):
        # Torch models return self; mirror that behavior.
        return self


def _load_model(kind: str, device: str, ckpt: str = None):
    if kind == "small":
        if ckpt:
            model = _CppPolicyAdapter(engine_path=ckpt)
        else:
            model = _CppPolicyAdapter(engine_path="model.trt")
    elif kind == "leela":
        model = LeelaBT4().to(device)
        model.load_state_dict(torch.load("src/models/leela/model.pt"))
        if ckpt:
            if hasattr(model, "load_state_dict"):
                print("found ckpt")
                sd = torch.load(ckpt, map_location=device, weights_only=False)
                state = sd["model_state_dict"]
                model.load_state_dict(state, strict=False)
            else:
                print(f"Warning: checkpoint {ckpt} provided but ignored for {kind} model.")
    else:
        raise ValueError(f"Unknown model kind: {kind}")

    if hasattr(model, "eval"):
        model = model.eval()
    return model


def main():
    p = argparse.ArgumentParser(description="Self-play arena for two players (models)")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--max-moves", type=int, default=512)
    p.add_argument("--player-1-temp", type=float, default=0.0)
    p.add_argument("--player-2-temp", type=float, default=0.0)
    p.add_argument("--target-games", type=int, default=None)
    p.add_argument("--target-hours", type=float, default=None)
    p.add_argument("--output-dir", type=str, default="data/selfplay_pgn")
    p.add_argument("--player-1-kind", type=str, default="small", choices=["small", "leela"]) 
    p.add_argument("--player-2-kind", type=str, default="leela", choices=["small", "leela"]) 
    p.add_argument("--player-1-ckpt", type=str, default=None)
    p.add_argument("--player-2-ckpt", type=str, default=None)
    p.add_argument("--player-1-device", type=str, default="cuda")
    p.add_argument("--player-2-device", type=str, default="cuda")
    p.add_argument("--player-1-fp16", action="store_true")
    p.add_argument("--player-2-fp16", action="store_true")
    p.add_argument("--swap-sides-every-n", type=int, default=0)
    p.add_argument("--player-1-name", type=str, default="player-1")
    p.add_argument("--player-2-name", type=str, default="player-2")
    p.add_argument("--randomize-starting-player", action="store_true")
    p.add_argument("--use-hf-openings", action="store_true")
    p.add_argument("--calibrate", action="store_true")
    p.add_argument("--calibration-games", type=int, default=512)
    p.add_argument("--calibration-mode", type=str, default="buckets", choices=["buckets", "second_order"])
    p.add_argument("--calibration-buckets", type=int, default=5)
    p.add_argument("--calibration-samples-per-bucket", type=int, default=256)
    p.add_argument("--calibration-temp-min", type=float, default=0.05)
    p.add_argument("--calibration-temp-max", type=float, default=2.0)

    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("loading player1 weights")
    player1_model = _load_model(args.player_1_kind, args.player_1_device, args.player_1_ckpt)
    print("loading player2 weights")
    player2_model = _load_model(args.player_2_kind, args.player_2_device, args.player_2_ckpt)

    cfg = SelfPlayConfig(
        batch_size=args.batch_size,
        max_moves=args.max_moves,
        player_1_temperature=args.player_1_temp,
        player_2_temperature=args.player_2_temp,
        device_player_1=args.player_1_device,
        device_player_2=args.player_2_device,
        player_1_fp16=args.player_1_fp16,
        player_2_fp16=args.player_2_fp16,
        target_games=args.target_games,
        target_hours=args.target_hours,
        output_dir=args.output_dir,
        swap_sides_every_n=args.swap_sides_every_n,
        player_1_name=args.player_1_name,
        player_2_name=args.player_2_name,
        randomize_starting_player=args.randomize_starting_player,
        fen_generator=(hf_opening_fen_generator if args.use_hf_openings else None),
        calibrate=args.calibrate,
        calibration_games=args.calibration_games,
        calibration_mode=args.calibration_mode,
        calibration_buckets=args.calibration_buckets,
        calibration_samples_per_bucket=args.calibration_samples_per_bucket,
        calibration_temp_min=args.calibration_temp_min,
        calibration_temp_max=args.calibration_temp_max,
    )

    arena = SelfPlayArena(player1_model, player2_model, cfg)
    run_ui(arena)


if __name__ == "__main__":
    main()
