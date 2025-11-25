import random
from typing import Iterator, Optional, Dict, Any

import bulletchess


def _load_hf_openings(split: str = "train"):

    from datasets import load_dataset  # type: ignore
    _DATASET = load_dataset("Lichess/chess-openings", split=split)
    return _DATASET


def _epd_to_fen(epd: str) -> Optional[str]:
    # EPD: fields similar to FEN but usually 4 tokens + operations; convert to FEN by taking first 4 and appending clocks
    parts = epd.strip().split()
    if len(parts) < 4:
        return None
    fen4 = " ".join(parts[:4])
    fen = f"{fen4} 0 1"
    try:
        # Validate
        _ = bulletchess.Board.from_fen(fen)
        return fen
    except Exception:
        return None


def hf_opening_fen_generator(split: str = "train", seed: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    """Yield random openings sampled from Lichess/chess-openings.

    Each item is a dict with:
      - 'fen': FEN string
      - 'uci': List[str] of UCI moves that lead to the FEN (may be empty)
    """
    ds = _load_hf_openings(split)
    rnd = random.Random(seed)
    n = len(ds)
    while True:
        idx = rnd.randrange(0, n)
        rec = ds[int(idx)]
        if not isinstance(rec, dict):
            continue
        epd = rec.get("epd")
        if not epd:
            continue
        fen = _epd_to_fen(epd)
        if fen is None:
            continue
        uci_str = rec.get("uci") or ""
        uci_moves = [mv for mv in uci_str.strip().split() if mv]
        yield {"fen": fen, "uci": uci_moves}

