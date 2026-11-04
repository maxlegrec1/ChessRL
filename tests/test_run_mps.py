from src.run import mps_child_env


def test_mps_child_env():
    base = {"PATH": "/bin", "CUDA_MPS_CLIENT_PRIORITY": "1"}
    env = mps_child_env(base, "/tmp/p", "/tmp/l")
    assert env["CUDA_MPS_PIPE_DIRECTORY"] == "/tmp/p"
    assert env["CUDA_MPS_LOG_DIRECTORY"] == "/tmp/l"
    assert env["PATH"] == "/bin"
    assert "CUDA_MPS_CLIENT_PRIORITY" not in env  # normal priority for self-play and the trainer
    assert base["CUDA_MPS_CLIENT_PRIORITY"] == "1"  # the caller's environment is untouched
