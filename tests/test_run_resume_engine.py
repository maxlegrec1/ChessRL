from src.run import _has_engine


def test_has_engine(tmp_path):
    assert not _has_engine(tmp_path)  # no engines/ at all (a branch copied in)
    (tmp_path / "engines").mkdir()
    it = tmp_path / "engines" / ".iteration"
    it.write_text("iteration,engine_path\n")
    assert not _has_engine(tmp_path)  # header only
    it.write_text("iteration,engine_path\n92500,/x/step_00092500.trt\n")
    assert _has_engine(tmp_path)
