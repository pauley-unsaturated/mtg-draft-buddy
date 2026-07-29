import draftbot


def test_package_imports():
    assert draftbot.__version__


def test_torch_available():
    import torch

    x = torch.arange(6).reshape(2, 3)
    assert int(x.sum()) == 15
