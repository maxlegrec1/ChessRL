from src.run import mps_child_env


def test_mps_child_env():
    base = {"PATH": "/bin", "CUDA_MPS_CLIENT_PRIORITY": "0"}
    sp = mps_child_env(base, "/tmp/p", "/tmp/l", trainer=False)
    tr = mps_child_env(base, "/tmp/p", "/tmp/l", trainer=True)
    for env in (sp, tr):
        assert env["CUDA_MPS_PIPE_DIRECTORY"] == "/tmp/p"
        assert env["CUDA_MPS_LOG_DIRECTORY"] == "/tmp/l"
        assert env["PATH"] == "/bin"
    assert "CUDA_MPS_CLIENT_PRIORITY" not in sp  # self-play at normal priority
    assert tr["CUDA_MPS_CLIENT_PRIORITY"] == "1"  # the trainer below normal
    assert base["CUDA_MPS_CLIENT_PRIORITY"] == "0"  # the caller's environment is untouched
