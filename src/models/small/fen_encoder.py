import bulletchess
import numpy as np
import torch
def fen_to_tensor(fen: str):
    board = bulletchess.Board.from_fen(fen)
    P = 19  # 12 planes for pieces + 1 for side to play + 1 for en passant + 4 for castling + 1 for 50-move rule
    tensor = np.zeros((8, 8, P), dtype=np.float32)
    
    # Populate 12 piece planes using fast bitboards
    type_order = [
        (bulletchess.PAWN, 0),
        (bulletchess.KNIGHT, 1),
        (bulletchess.BISHOP, 2),
        (bulletchess.ROOK, 3),
        (bulletchess.QUEEN, 4),
        (bulletchess.KING, 5),
    ]
    for color in (bulletchess.WHITE, bulletchess.BLACK):
        base = 0 if color == bulletchess.WHITE else 6
        for piece_type, offset in type_order:
            bitboard = board[color, piece_type]
            for sq in bitboard:
                idx = sq.index()
                rank = idx // 8
                file = idx % 8
                tensor[7 - rank, file, base + offset] = 1.0
    
    # Side to play plane
    tensor[:, :, 12] = 1.0 if board.turn == bulletchess.WHITE else 0.0

    # En passant plane
    if board.en_passant_square is not None:
        ep_idx = board.en_passant_square.index()
        rank = ep_idx // 8
        file = ep_idx % 8
        tensor[7 - rank, file, 13] = 1.0
    
    # Castling rights planes (4 total: white kingside, white queenside, black kingside, black queenside)
    tensor[:, :, 14] = 1.0 if (bulletchess.WHITE_KINGSIDE in board.castling_rights) else 0.0
    tensor[:, :, 15] = 1.0 if (bulletchess.WHITE_QUEENSIDE in board.castling_rights) else 0.0
    tensor[:, :, 16] = 1.0 if (bulletchess.BLACK_KINGSIDE in board.castling_rights) else 0.0
    tensor[:, :, 17] = 1.0 if (bulletchess.BLACK_QUEENSIDE in board.castling_rights) else 0.0
    
    # 50-move rule plane (normalized to [0,1])
    tensor[:, :, 18] = min(board.halfmove_clock / 100.0, 1.0)
    
    return tensor


def boards_to_tensor(boards, device: str = "cuda", dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Convert a list of bulletchess.Board into a batched tensor of shape [B, 8, 8, 19].
    This avoids reparsing FEN strings and reuses existing board objects.
    """
    if not boards:
        return torch.empty((0, 8, 8, 19), device=device, dtype=dtype)
    B = len(boards)
    out = np.zeros((B, 8, 8, 19), dtype=np.float32)
    type_order = [
        (bulletchess.PAWN, 0),
        (bulletchess.KNIGHT, 1),
        (bulletchess.BISHOP, 2),
        (bulletchess.ROOK, 3),
        (bulletchess.QUEEN, 4),
        (bulletchess.KING, 5),
    ]
    for b_idx, board in enumerate(boards):
        # Pieces
        for color in (bulletchess.WHITE, bulletchess.BLACK):
            base = 0 if color == bulletchess.WHITE else 6
            for piece_type, offset in type_order:
                bitboard = board[color, piece_type]
                for sq in bitboard:
                    idx = sq.index()
                    rank = idx // 8
                    file = idx % 8
                    out[b_idx, 7 - rank, file, base + offset] = 1.0
        # Side to move
        out[b_idx, :, :, 12] = 1.0 if board.turn == bulletchess.WHITE else 0.0
        # En passant
        if board.en_passant_square is not None:
            ep_idx = board.en_passant_square.index()
            rank = ep_idx // 8
            file = ep_idx % 8
            out[b_idx, 7 - rank, file, 13] = 1.0
        # Castling rights
        out[b_idx, :, :, 14] = 1.0 if (bulletchess.WHITE_KINGSIDE in board.castling_rights) else 0.0
        out[b_idx, :, :, 15] = 1.0 if (bulletchess.WHITE_QUEENSIDE in board.castling_rights) else 0.0
        out[b_idx, :, :, 16] = 1.0 if (bulletchess.BLACK_KINGSIDE in board.castling_rights) else 0.0
        out[b_idx, :, :, 17] = 1.0 if (bulletchess.BLACK_QUEENSIDE in board.castling_rights) else 0.0
        # 50-move counter
        out[b_idx, :, :, 18] = min(board.halfmove_clock / 100.0, 1.0)
    return torch.from_numpy(out).to(device=device, dtype=dtype)


def batch_fen_to_tensor(fens, device: str = "cuda", dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Batched FEN encoder. Returns [B, 8, 8, 19] on the specified device/dtype.
    Uses preallocation to reduce Python overhead.
    """
    if not fens:
        return torch.empty((0, 8, 8, 19), device=device, dtype=dtype)
    B = len(fens)
    out = np.zeros((B, 8, 8, 19), dtype=np.float32)
    for i, fen in enumerate(fens):
        out[i] = fen_to_tensor(fen)
    return torch.from_numpy(out).to(device=device, dtype=dtype)


def apply_move_to_tensor(board_tensor: torch.Tensor, move_uci: str, board: "bulletchess.Board") -> torch.Tensor:
    """
    Return a new tensor for the position after applying `move_uci` to `board`.
    For correctness and simplicity, this recomputes the full tensor from the updated board.
    """
    mv = bulletchess.Move.from_uci(move_uci)
    board.apply(mv)
    try:
        new_tensor = boards_to_tensor([board], device=board_tensor.device.type, dtype=board_tensor.dtype)
        # boards_to_tensor returns [1,8,8,19]
        return new_tensor[0]
    finally:
        board.undo()
