from src.run import _has_engine


def test_has_engine(tmp_path):
    assert not _has_engine(tmp_path)  # no engines/ at all (a branch copied in)
    (tmp_path / "engines").mkdir()
    it = tmp_path / "engines" / ".iteration"
    it.write_text("iteration,engine_path\n")
    assert not _has_engine(tmp_path)  # header only
    it.write_text("iteration,engine_path\n92500,/x/step_00092500.cnet\n")
    assert not _has_engine(tmp_path)  # the listed network is missing
    net = tmp_path / "engines" / "step_00092500.cnet"
    net.write_bytes(b"")
    it.write_text(f"iteration,engine_path\n92500,{net}\n")
    assert _has_engine(tmp_path)
    # A run from before the fused network: its last engine is a TensorRT one.
    trt = tmp_path / "engines" / "step_00093000.trt"
    trt.write_bytes(b"")
    it.write_text(f"iteration,engine_path\n92500,{net}\n93000,{trt}\n")
    assert not _has_engine(tmp_path)
