import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.models.resnet import Bottleneck


def build_backbone(weights: ResNet50_Weights | None = None) -> nn.Sequential:
    """ResNet-50 up to layer4 with last stride 1. Returns (B, 2048, H/16, W/16)."""

    resnet = resnet50(weights=weights)

    block = resnet.layer4[0]
    assert isinstance(block, Bottleneck)
    assert isinstance(block.downsample, nn.Sequential)
    shortcut_conv = block.downsample[0]
    assert isinstance(shortcut_conv, nn.Conv2d)
    block.conv2.stride = (1, 1)
    shortcut_conv.stride = (1, 1)

    return nn.Sequential(
        resnet.conv1,
        resnet.bn1,
        resnet.relu,
        resnet.maxpool,
        resnet.layer1,
        resnet.layer2,
        resnet.layer3,
        resnet.layer4,
    )


class SpatialAttention(nn.Module):

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=2, 
            out_channels=1, 
            kernel_size=kernel_size, 
            padding=kernel_size//2, 
            bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _avg = x.mean(dim=1, keepdim=True)
        _max = x.amax(dim=1, keepdim=True)

        mask = torch.sigmoid(self.conv(torch.concat([_avg, _max], dim=1)))

        return x * mask

