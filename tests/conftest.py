import shutil

import pytest


def pytest_collection_modifyitems(config, items):
    try:
        import torch
        has_gpu = torch.cuda.is_available()
    except Exception:
        has_gpu = False
    has_trt = shutil.which("trtexec") is not None

    skip_gpu = pytest.mark.skip(reason="no CUDA")
    skip_trt = pytest.mark.skip(reason="no trtexec")
    for item in items:
        if "gpu" in item.keywords and not has_gpu:
            item.add_marker(skip_gpu)
        if "trt" in item.keywords and not has_trt:
            item.add_marker(skip_trt)
