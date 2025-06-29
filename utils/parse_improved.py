import os
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from utils.vocab import policy_index
from utils.fen_encoder import fen_to_tensor

# Precompute mapping from move string to index for fast lookup
_policy_map = {uci: idx for idx, uci in enumerate(policy_index)}
_pad_id = len(policy_index)-1  # fallback padding ID

input_dir = "parquet_chunks"
block_size = 128
clip_length = block_size - 64

def encode_fens(fen_list):
    """
    Encode a list of FEN strings into a batched tensor.
    """
    # fen_to_tensor returns numpy.ndarray, so convert to torch.Tensor
    tensors = [torch.from_numpy(fen_to_tensor(fen)) for fen in fen_list]
    return torch.stack(tensors)


def encode_moves(moves_list):
    """
    Map UCI move strings to their indices, with fallback by stripping the last character.
    """
    ids = []
    for mov in moves_list:
        if mov in _policy_map:
            ids.append(_policy_map[mov])
        else:
            ids.append(_policy_map.get(mov[:-1], _pad_id))
    return torch.tensor(ids, dtype=torch.long)


def clip_and_batch(moves_tensors, clip=clip_length):
    """
    Pad and/or clip a list of move-index tensors to shape (batch_size, clip).
    """
    padded = pad_sequence(moves_tensors, batch_first=True, padding_value=_pad_id)
    if padded.size(1) >= clip:
        return padded[:, :clip]
    extra = padded.new_full((padded.size(0), clip - padded.size(1)), _pad_id)
    return torch.cat([padded, extra], dim=1)


def batch_generator(input_dir, batch_size, return_fen=False, triple=False, device='cuda'):
    """
    Yield batches of encoded fens and moves from parquet files.

    Args:
        input_dir (str): Directory containing .parquet files.
        batch_size (int): Number of games per batch.
        return_fen (bool): If True, yield raw FEN list only.
        triple (bool): If True, also return raw FEN strings in addition to tensors.
        device (str): Target device for tensors.
    """
    files = sorted([os.path.join(input_dir, f)
                    for f in os.listdir(input_dir) if f.endswith('.parquet')])

    fen_batch, move_batch = [], []
    for path in files:
        df = pd.read_parquet(path)
        for fen, start_idx, moves in df.itertuples(index=False, name=None):
            fen_batch.append(fen)
            move_batch.append(encode_moves(moves))
            if len(fen_batch) == batch_size:
                yield from _yield_batch(fen_batch, move_batch, return_fen, triple, device)
                fen_batch.clear()
                move_batch.clear()

    # Yield final partial batch if any
    if fen_batch:
        yield from _yield_batch(fen_batch, move_batch, return_fen, triple, device)


def _yield_batch(fen_batch, move_batch, return_fen, triple, device):
    if return_fen:
        yield fen_batch
    else:
        fens = encode_fens(fen_batch).to(device)
        moves_b = clip_and_batch(move_batch).to(device)
        if triple:
            yield fens, moves_b, fen_batch
        else:
            yield fens, moves_b


if __name__ == '__main__':
    from tqdm import tqdm
    gen = batch_generator(input_dir, batch_size=400)
    for _ in tqdm(range(1000)):
        batch = next(gen)
        print(batch[0].shape, batch[1].shape)
