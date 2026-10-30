import pytest


def pytest_collection_modifyitems(config, items):
    try:
        import torch

        has_gpu = torch.cuda.is_available()
    except Exception:
        has_gpu = False

    skip_gpu = pytest.mark.skip(reason="no CUDA")
    for item in items:
        if "gpu" in item.keywords and not has_gpu:
            item.add_marker(skip_gpu)
