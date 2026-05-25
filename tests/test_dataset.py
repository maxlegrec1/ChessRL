from src.train.dataset import SelfPlayDataset


def test_empty_buffer(tmp_path):
    p = tmp_path / "buf.bin"
    p.write_bytes(b"")
    ds = SelfPlayDataset(p, reload_seconds=3600)
    assert ds._size == 0
