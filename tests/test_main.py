import torch

from lab03.main import get_device


def test_get_device_returns_torch_device():
    assert isinstance(get_device(), torch.device)
