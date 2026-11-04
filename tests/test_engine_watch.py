from src.self_play.engine_watch import EngineWatcher


def test_pickup_latest_row(tmp_path):
    f = tmp_path / ".iteration"
    f.write_text("iteration,engine_path\n100,/tmp/a.cnet\n200,/tmp/b.cnet\n")
    w = EngineWatcher(f, poll_interval=0)
    assert w.engine_path == "/tmp/b.cnet"


def test_missing_file_keeps_empty(tmp_path):
    f = tmp_path / "absent.iteration"
    w = EngineWatcher(f, poll_interval=0)
    # No fallback: the watcher idles with an empty path until .iteration exists.
    assert w.engine_path == ""
