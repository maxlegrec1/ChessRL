"""Muon (src/train/muon.py): the orthogonalisation, one step on the encoder, the overflow skip, the state round
trip, and the trainer with optimizer: muon."""

import pytest
import torch

from src.models.encoder import EncoderModel
from src.train.muon import MuonAdamW, orthogonalize

from .test_trainer_smoke import _config


def test_orthogonalize_brings_singular_values_near_one():
    g = torch.Generator().manual_seed(0)
    for shape in ((4, 128, 128), (3, 256, 128), (2, 128, 256)):
        # Singular values over two decades: after the Frobenius scaling they land in [1e-3, 0.3], the range the
        # Polar Express coefficients are tuned for.
        u, _, vh = torch.linalg.svd(torch.randn(*shape, generator=g), full_matrices=False)
        s = torch.logspace(-2, 0, u.shape[-1])
        x = orthogonalize(u @ torch.diag_embed(s.expand(shape[0], -1)) @ vh * 7.0)
        sv = torch.linalg.svdvals(x)
        assert sv.min() > 0.7 and sv.max() < 1.25, (shape, sv.min(), sv.max())


def _optimizer(model, lr=1e-3, muon_lr=0.01):
    matrices = model.muon_matrices()
    on_muon = {id(p) for p, _ in matrices}
    rest = [p for p in model.parameters() if id(p) not in on_muon]
    return MuonAdamW(matrices, [{"params": rest}], lr=lr, betas=(0.9, 0.95), muon_lr=muon_lr,
                     muon_weight_decay=0.01)  # fmt: skip


def _loss(model, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(4, 8, 8, 111, generator=g)
    p, v, *_ = model(x, return_all=True)
    return p.logsumexp(-1).mean() + v.square().mean()


def test_step_moves_both_halves_and_skips_on_overflow():
    torch.manual_seed(0)
    model = EncoderModel()
    opt = _optimizer(model)
    assert len(opt.muon_group["params"]) == 16
    assert opt.muon_group["lr_scale"] == pytest.approx(10.0)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    _loss(model).backward()
    opt.step()
    moved = {n for n, p in model.named_parameters() if not torch.equal(p, before[n])}
    assert "blocks.0.qkv.weight" in moved and "blocks.3.down.weight" in moved  # Muon
    assert "embed.weight" in moved and "pol_q.weight" in moved  # AdamW
    # A Muon update has the size of an orthogonal matrix: lr * scale per singular value.
    d = (model.blocks[0].out.weight - before["blocks.0.out.weight"]).detach()
    d = d + 0.01 * 0.01 * before["blocks.0.out.weight"]  # undo the weight decay
    assert torch.linalg.svdvals(d).max() == pytest.approx(0.01, rel=0.3)

    # found_inf set by the loss scaler: nothing changes, not even the momentum buffers.
    state = {n: p.detach().clone() for n, p in model.named_parameters()}
    bufs = [b.clone() for b in opt.bufs]
    opt.zero_grad()
    _loss(model, seed=1).backward()
    for p in model.parameters():
        if p.grad is not None:
            p.grad.view(-1)[0] = float("inf")
    opt.found_inf = torch.ones(())
    with torch.no_grad():
        opt._muon_step(opt.found_inf, None)  # the AdamW half on the CPU has no GPU skip path
    del opt.found_inf
    for name, p in model.named_parameters():
        if any(p is q for q in opt.muon_group["params"]):
            assert torch.equal(p, state[name]), name
    assert all(torch.equal(a, b) for a, b in zip(bufs, opt.bufs, strict=True))


def test_state_dict_round_trip():
    torch.manual_seed(0)
    model = EncoderModel()
    opt = _optimizer(model)
    _loss(model).backward()
    opt.step()
    other = _optimizer(model)
    other.load_state_dict(opt.state_dict())
    assert all(torch.equal(a, b) for a, b in zip(opt.bufs, other.bufs, strict=True))
    with pytest.raises(ValueError):
        _optimizer(EncoderModel(layers=2)).load_state_dict(opt.state_dict())


def test_trainer_with_muon(tmp_path):
    cfg = _config(tmp_path, "visits")
    cfg.growth.schedule = []
    cfg.model.arch = "encoder"
    cfg.optimizer = "muon"
    cfg.muon_lr = 0.006
    trainer = Trainer(cfg)
    assert isinstance(trainer.optimizer, MuonAdamW)
    losses = []
    for _ in range(3):
        batch = trainer.buffer.sample_batch(cfg.batch_size)
        metrics = trainer._train_step(batch, expensive=False, log=True)
        losses.append(float(metrics["loss/total"]))
        trainer.step += 1
    assert all(torch.isfinite(torch.tensor(losses)))
    assert trainer.optimizer.muon_group["lr"] == pytest.approx(trainer.lr_at(2) * 6.0)
    trainer._save_checkpoint()
    cfg.resume_path = sorted(cfg.checkpoint.dir.glob("step_*.pt"))[-1]
    resumed = Trainer(cfg)
    assert all(torch.equal(a, b) for a, b in zip(trainer.optimizer.bufs, resumed.optimizer.bufs, strict=True))


from src.train.trainer import Trainer  # noqa: E402
