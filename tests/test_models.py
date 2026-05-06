import torch

from src.models.small.small_model import SmallModel


def test_small_forward_policy_only():
    m = SmallModel().eval()
    x = torch.zeros(2, 8, 8, 19)
    with torch.no_grad():
        policy = m(x)
    assert policy.shape == (2, 1929)


def test_small_forward_return_all():
    m = SmallModel().eval()
    x = torch.zeros(2, 8, 8, 19)
    with torch.no_grad():
        policy, value, value_q, mlh, mlh_q = m(x, return_all=True)
    assert policy.shape == (2, 1929)
    assert value.shape == (2, 3)
    assert value_q.shape == (2, 3)
    assert mlh.shape == (2,)
    assert mlh_q.shape == (2,)
