import zstandard as zstd
import pickle
import torch
import os
import random
def load_batch(file_path, device="cuda"):
    """
    Load a compressed and pickled batch file.
    
    Args:
        file_path (str): Path to the .pkl.zst batch file.
        device (str): Device to move tensors to ("cuda" or "cpu").
    
    Returns:
        Tuple of tensors: (fens, moves, policies)
    """
    with open(file_path, 'rb') as f:
        compressed = f.read()
    decompressed = zstd.ZstdDecompressor().decompress(compressed)
    encoded_fens, moves, policies,fens = pickle.loads(decompressed)
    return encoded_fens.to(device), moves.to(device), policies.to(device),fens



def dir_iterator(dir_path,device = "cuda"):
    if isinstance(dir_path, str):
        files = os.listdir(dir_path)
        files = [os.path.join(dir_path, file) for file in files]
    if isinstance(dir_path, list):
        print("many dirs")
        files = []
        for path in dir_path:
            gothrough = os.listdir(path)
            print("path : ",path, "files : ",len(gothrough))
            for file in gothrough:
                files.append(os.path.join(path, file))
    files = [f for f in files if f.endswith('.pkl.zst')]
    random.shuffle(files)
    print(f"Found {len(files)} files in {dir_path}")
    for file in files:
        #file_path = os.path.join(dir_path, file)
        #print(f"Loading {file_path}")
        encoded_fens, moves, policies,fens = load_batch(file, device=device)
        yield [encoded_fens[:110], moves[:110]], policies[:110],fens[:110]


if __name__ == "__main__":
    from parse import policy_index
    device = "cpu"

    path = "data/preprocessed/batch_12000.pkl.zst"

    # Load the batch
    encoded_fens, moves, policies, fens = load_batch(path, device=device)
    
    print(f"Encoded FENs: {encoded_fens.shape}")
    print(f"Moves: {moves.shape}")
    print(f"Policies: {policies.shape}")
    print(f"FENs: {len(fens)}")


    print("first_fen : ",fens[0])
    print("moves : ",moves[0])

    for move in moves[0][:9]:
        print(policy_index[move.item()])



    last_policy = policies[0][8]
    print("last_policy shape : ",last_policy.shape)

    k = 10

    values, indices = torch.topk(last_policy, k=k)
    for i in range(k):
        print(f"Move {i}: {policy_index[indices[i].item()]} with value {values[i].item()}")

    print(policies[0][0])