import chess
import torch

from src.models.encoder import EncoderModel, _token
from src.models.factory import build_model, model_from_checkpoint
from src.models.small.vocab import policy_to_idx


def test_shapes_and_spec():
    m = build_model({"arch": "encoder", "d_model": 64, "layers": 2, "heads": 4, "d_ff": 128})
    out = m(torch.rand(3, 8, 8, 111), return_all=True)
    assert [tuple(o.shape) for o in out] == [(3, 1929), (3, 3), (3, 3), (3,), (3,)]
    assert m.spec()["arch"] == "encoder"
    m2 = model_from_checkpoint({"model": m.state_dict(), "model_spec": m.spec()})
    x = torch.rand(2, 8, 8, 111)
    torch.testing.assert_close(m.eval()(x), m2.eval()(x))


def test_policy_logit_uses_from_and_to_squares():
    # e2e4 must read the query at e2's token and the key at e4's token.
    m = EncoderModel(d_model=32, layers=1, heads=4, d_ff=64)
    i = policy_to_idx["e2e4"]
    assert int(m.v_from[i]) == _token(chess.E2) and int(m.v_to[i]) == _token(chess.E4)
    assert _token(chess.A1) == 56 and _token(chess.H8) == 7


def test_policy_head_matches_naive_formula():
    torch.manual_seed(0)
    m = EncoderModel(d_model=32, layers=1, heads=4, d_ff=64).eval()
    with torch.no_grad():
        m.promo.normal_()
    x = torch.rand(2, 8, 8, 111)
    logits = m(x)
    # Re-derive with the per-entry gather form.
    h = m.embed(x.reshape(2, 64, 111)) + m.pos
    x0 = h
    for blk in m.blocks:
        h = blk(h, x0, m.pair_bias)
    h = m.norm(h)
    q, k = m.pol_q(h), m.pol_k(h)
    naive = ((q[:, m.v_from] + m.promo[m.v_promo]) * k[:, m.v_to]).sum(-1) / 32**0.5
    real = ~m.v_special
    torch.testing.assert_close(logits[:, real], naive[:, real], rtol=1e-4, atol=1e-5)


def test_onnx_export(tmp_path):
    m = EncoderModel(d_model=32, layers=1, heads=4, d_ff=64).eval()

    class Wrap(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            p, _v, vq, mlh, mlh_q = self.inner(x, return_all=True)
            return p, torch.softmax(vq, dim=-1), mlh, mlh_q

    torch.onnx.export(Wrap(m), torch.rand(2, 8, 8, 111), (tmp_path / "enc.onnx").as_posix(),
                      input_names=["input"], output_names=["policy", "value", "mlh", "mlh_q"],
                      dynamic_axes={"input": {0: "batch_size"}}, opset_version=17, dynamo=False)
    assert (tmp_path / "enc.onnx").stat().st_size > 0
