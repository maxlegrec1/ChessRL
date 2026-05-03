from src.self_play.engine_watch import EngineWatcher


def test_pickup_latest_row(tmp_path):
    f = tmp_path / ".iteration"
    f.write_text("iteration,engine_path\n100,/tmp/a.trt\n200,/tmp/b.trt\n")
    w = EngineWatcher(f, poll_interval=0)
    assert w.engine_path == "/tmp/b.trt"


def test_missing_file_keeps_default(tmp_path):
    f = tmp_path / "absent.iteration"
    w = EngineWatcher(f, poll_interval=0)
    # default fallback when the file doesn't exist
    assert w.engine_path == "artifacts/engines/model_dynamic_initial.trt"
