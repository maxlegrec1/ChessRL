from src.models.small.vocab import canonical_key, mirror_uci, policy_index, policy_to_idx


def test_size():
    assert len(policy_index) == 1858  # white's moves (lc0's set)
    assert len(policy_to_idx) == len(policy_index)


def test_lookup_consistent():
    for sample in ("e2e4", "g1f3", "a1a2", "e1g1", "a7a8q"):
        idx = policy_to_idx[sample]
        assert policy_index[idx] == sample
        assert 0 <= idx < len(policy_index)


def test_canonical_keys():
    assert mirror_uci("e7e5") == "e2e4" and mirror_uci("a2a1q") == "a7a8q"
    assert canonical_key("e2e4", True) == "e2e4"
    assert canonical_key("e7e5", False) == "e2e4"
    assert canonical_key("e8g8", False) == "e1g1"  # black castles as white does
    assert canonical_key("b2b1n", False) == "b7b8"  # knight promotion: the bare move
