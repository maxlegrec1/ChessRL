import torch   
import time
import torch_tensorrt
from src.models.small import Inference
from src.models.small.fen_encoder import fen_to_tensor, batch_fen_to_tensor
from src.models.small.vocab import policy_index,policy_to_idx
import torch.nn.functional as F
import bulletchess
model = Inference().to("cuda").eval()
model.load_state_dict(torch.load("src/models/small/model.pt"),strict=False)
fen = "r3k2r/ppp2ppp/2P5/8/2p1q1b1/B1P2N2/P4PPP/R2Q2K1 b kq - 0 14"

inputs = [torch.randn((1024, 8, 8, 19), device="cuda", dtype=torch.float16)]
# trt_gm is a torch.fx.GraphModule object
print("Starting TensorRT compilation...")
trt_gm = torch_tensorrt.compile(
    model.eval(),
    ir="dynamo",
    arg_inputs=inputs,
    enabled_precisions={torch_tensorrt.dtype.half},
    require_full_compilation=True,
)
print("TensorRT compilation completed!")
print("Saving compiled model...", flush=True)
save_start = time.time()
torch_tensorrt.save(trt_gm, "trt.ts", output_format="torchscript", arg_inputs=inputs)
save_end = time.time()
print(f"Model saved! (took {save_end - save_start:.2f} seconds)")



# Try loading and testing the saved model
print("\nTesting inference with loaded model...")


trt_model = torch.jit.load("trt.ts").cuda()


class WrappedModel():
    def __init__(self,):
        self.model = trt_model

    def __call__(self,fens, T = 0, return_probs = False, force_legal = True):
        import concurrent.futures as _futures
        def _sync():
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        timings: list[tuple[str, float]] = []
        total_start = time.perf_counter()

        encode_start = time.perf_counter()
        device = "cuda"
        batch_x = batch_fen_to_tensor(fens, device=device, dtype=torch.float16)
        encode_end = time.perf_counter()
        timings.append(("encode+stack", encode_end - encode_start))

        print(f"Prepared batch input shape: {batch_x.shape}")
        # Kick off CPU legality mask build to overlap with GPU inference
        legal_future = None
        if force_legal:
            def _build_legal_mask():
                mask = torch.full((len(fens), 1929), -999.0, device=device, dtype=torch.float16)
                for i, fen in enumerate(fens):
                    board = bulletchess.Board.from_fen(fen)
                    for mv in board.legal_moves():
                        u = mv.uci()
                        if u[-1] == 'n':
                            u = u[:-1]
                        idx = policy_to_idx.get(u)
                        if idx is not None:
                            mask[i, idx] = 0.0
                return mask
            ex = _futures.ThreadPoolExecutor(max_workers=1)
            legal_future = ex.submit(_build_legal_mask)

        with torch.no_grad():
            _sync()
            inference_start = time.perf_counter()
            logits = self.model(batch_x)
            _sync()
            inference_end = time.perf_counter()
        timings.append(("inference", inference_end - inference_start))

        if force_legal:
            legal_start = time.perf_counter()
            legal_mask = legal_future.result() if legal_future is not None else None
            logits = logits + legal_mask
            _sync()
            legal_end = time.perf_counter()
            timings.append(("legal_mask", legal_end - legal_start))

        moves: list[str] = []
        probs_tensor: torch.Tensor
        select_start = time.perf_counter()
        if T == 0.0:
            best_idx = torch.argmax(logits, dim=1)
            for idx in best_idx.tolist():
                moves.append(policy_index[idx])
            if return_probs:
                probs_tensor = torch.zeros_like(logits)
                probs_tensor[torch.arange(1024), best_idx] = 1.0
        else:
            if return_probs:
                scaled = logits / T
                probs_tensor = F.softmax(scaled, dim=1)
                sampled = torch.multinomial(probs_tensor, num_samples=1).squeeze(1)
                for idx in sampled.tolist():
                    moves.append(policy_index[idx])
            else:
                # Gumbel-max sampling with top-k restriction for speed
                topk_vals, topk_idx = torch.topk(logits, k=128, dim=1)
                g = -torch.log(-torch.log(torch.rand_like(topk_vals)))
                sampled_local = torch.argmax(topk_vals / T + g, dim=1)
                sampled = topk_idx.gather(1, sampled_local.unsqueeze(1)).squeeze(1)
                for idx in sampled.tolist():
                    moves.append(policy_index[idx])
        _sync()
        select_end = time.perf_counter()
        timings.append(("select_move", select_end - select_start))

        total_end = time.perf_counter()
        timings.append(("total", total_end - total_start))

        print("Timing breakdown:")
        for name, duration in timings:
            print(f" - {name}: {duration:.4f} seconds")

        # Correctness checks
        if force_legal:
            for i, mv in enumerate(moves):
                brd = bulletchess.Board.from_fen(fens[i])
                legal_uci = set(m.uci().rstrip('n') for m in brd.legal_moves())
                assert mv in legal_uci, f"Selected illegal move {mv} for row {i}"

        return moves




wrapped = WrappedModel()

print(wrapped([fen]*1024, T = 0.1))