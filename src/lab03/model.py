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
