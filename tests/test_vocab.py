from src.models.small.vocab import policy_index, policy_to_idx


def test_size():
    assert len(policy_index) == 1929
    assert len(policy_to_idx) == len(policy_index)


def test_lookup_consistent():
    for sample in ("e2e4", "g1f3", "a1a2"):
        idx = policy_to_idx[sample]
        assert policy_index[idx] == sample
        assert 0 <= idx < len(policy_index)
