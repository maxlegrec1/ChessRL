import torch

from src.models.factory import build_model


def test_default_model_forward():
    m = build_model().eval()  # the recipe's encoder: d_model 128, 4 layers
    x = torch.zeros(2, 8, 8, 111)
    with torch.no_grad():
        policy, value, value_q, mlh, mlh_q = m(x, return_all=True)
        assert m(x).shape == (2, 1858)
    assert policy.shape == (2, 1858)
    assert value.shape == (2, 3) and value_q.shape == (2, 3)
    assert mlh.shape == (2,) and mlh_q.shape == (2,)
