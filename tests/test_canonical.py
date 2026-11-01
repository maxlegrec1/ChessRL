"""Canonical boards (EncoderModel(canonical=True)): a position and its colour-flipped twin are the same input to
the network, its outputs fill exactly the canonical moves, and training, growth and checkpoints carry the flag."""

import chess
import numpy as np
import torch

from src.analysis.engine import encode, policy_key
from src.models.encoder import ABSENT_LOGIT, CANONICAL_VOCAB, EncoderModel
from src.models.growth import grow_encoder
from src.models.small.vocab import policy_to_idx
from src.train.dataset import MIRROR_INDEX

from .test_trainer_smoke import _config

FENS = [
    chess.STARTING_FEN,
    # Castling rights on one side only, an en passant square, a pawn that can promote by capture or push.
    "r3k2r/pPp2ppp/8/3pP3/8/8/P1PP1PPP/R3K1NR w Kq d6 0 12",
    # Black to move, a black pawn about to promote.
    "8/2P1k3/8/8/8/8/4K1p1/8 b - - 3 40",
]


def _model():
    torch.manual_seed(0)
    model = EncoderModel(canonical=True).eval()
    with torch.no_grad():  # make the parameters that start at zero matter
        model.pair_bias.normal_(0, 1.0)
        model.promo.normal_(0, 0.5)
    return model


def _run(model, boards):
    x = torch.from_numpy(np.stack([encode(b) for b in boards]))
    with torch.no_grad():
        return model(x, return_all=True)


def test_vocabulary():
    assert len(CANONICAL_VOCAB) == 1858
    assert not any(len(m) == 5 and m[3] == "1" for m in CANONICAL_VOCAB)


def test_colour_flipped_twin_is_the_same_input():
    model = _model()
    boards = [chess.Board(f) for f in FENS]
    out = _run(model, boards)
    twin = _run(model, [b.mirror() for b in boards])
    for a, b in zip(out[1:], twin[1:], strict=True):  # value, value_q, moves left: the side to move's view
        assert torch.equal(a, b)
    # The policy: move i in a position is move MIRROR_INDEX[i] in its twin.
    assert torch.equal(out[0], twin[0][:, torch.from_numpy(MIRROR_INDEX)])


def test_policy_fills_the_canonical_moves_and_every_legal_move():
    model = _model()
    boards = [chess.Board(f) for f in FENS]
    boards += [b.mirror() for b in boards]
    logits = _run(model, boards)[0]
    assert ((logits != ABSENT_LOGIT).sum(1) == len(CANONICAL_VOCAB)).all()
    for row, board in zip(logits, boards, strict=True):
        for move in board.legal_moves:
            assert row[policy_to_idx[policy_key(move)]] != ABSENT_LOGIT, (board.fen(), move)


def test_absolute_network_unchanged():
    model = EncoderModel()
    assert model.spec()["canonical"] is False
    assert hasattr(model, "special_logit")


def test_growth_keeps_canonical():
    grown = grow_encoder(_model(), 6)
    assert grown.cfg["canonical"] is True and grown.cfg["layers"] == 6


def test_trainer_with_canonical_encoder(tmp_path):
    from src.train.trainer import Trainer

    cfg = _config(tmp_path, "visits")
    cfg.growth.schedule = []
    cfg.model.arch = "encoder"
    cfg.model.canonical = True
    trainer = Trainer(cfg)
    for _ in range(2):
        batch = trainer.buffer.sample_batch(cfg.batch_size, mirror_prob=0.5)
        metrics = trainer._train_step(batch, expensive=True)
        assert np.isfinite(float(metrics["loss/total"]))
        assert 0.0 <= metrics["legal_move_prob"] <= 1.0
        trainer.step += 1
    trainer._save_checkpoint()
    cfg.resume_path = sorted(cfg.checkpoint.dir.glob("step_*.pt"))[-1]
    assert Trainer(cfg).model.spec()["canonical"] is True


def test_mirror_index_is_an_involution():
    # The twin check relies on it: the twin of the twin is the position itself.
    assert (MIRROR_INDEX[MIRROR_INDEX] == np.arange(len(MIRROR_INDEX))).all()
