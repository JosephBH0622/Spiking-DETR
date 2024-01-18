import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List

from spikingjelly.activation_based import layer, neuron

from util_my.misc_my import NestedTensor

from models_my.position_encoding_my import build_position_encoding


class SpikingBTNK1(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(SpikingBTNK1, self).__init__()
        self.stem = nn.Sequential(
            neuron.ParametricLIFNode(backend='cupy'),
            layer.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), stride=stride),
            layer.BatchNorm2d(out_channels),

            neuron.ParametricLIFNode(backend='cupy'),
            layer.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), padding=1,
                         stride=(1, 1)),
            layer.BatchNorm2d(out_channels),

            neuron.ParametricLIFNode(backend='cupy'),
            layer.Conv2d(in_channels=out_channels, out_channels=out_channels * 4, kernel_size=(1, 1), stride=(1, 1)),
            layer.BatchNorm2d(out_channels * 4)
        )

        self.res = nn.Sequential(
            neuron.ParametricLIFNode(backend='cupy'),
            layer.Conv2d(in_channels=in_channels, out_channels=out_channels * 4, kernel_size=(1, 1), stride=stride),
            layer.BatchNorm2d(out_channels * 4)
        )

    def forward(self, x):
        out = x
        out_stem = self.stem(out)
        out_res = self.res(out)
        out = out_res + out_stem
        return out


class SpikingBTNK2(nn.Module):
    def __init__(self, channels):
        super(SpikingBTNK2, self).__init__()
        self.stem = nn.Sequential(
            neuron.ParametricLIFNode(backend='cupy'),
            layer.Conv2d(in_channels=channels, out_channels=channels // 4, kernel_size=(1, 1), stride=(1, 1)),
            layer.BatchNorm2d(channels // 4),

            neuron.ParametricLIFNode(backend='cupy'),
            layer.Conv2d(in_channels=channels // 4, out_channels=channels // 4, kernel_size=(3, 3), padding=1,
                         stride=(1, 1)),
            layer.BatchNorm2d(channels // 4),

            neuron.ParametricLIFNode(backend='cupy'),
            layer.Conv2d(in_channels=channels // 4, out_channels=channels, kernel_size=(1, 1), stride=(1, 1)),
            layer.BatchNorm2d(channels)
        )

    def forward(self, x):
        out = x
        out_stem = self.stem(out)
        out = out + out_stem
        return out


# SpikingResNet的Backbone
class NewSpikingResNet(nn.Module):
    def __init__(self):
        super(NewSpikingResNet, self).__init__()
        self.stage_0 = nn.Sequential(
            layer.Conv2d(in_channels=2, out_channels=64, kernel_size=(7, 7), stride=(2, 2)),
            layer.BatchNorm2d(64)
        )

        self.stage_1 = nn.Sequential(
            SpikingBTNK1(in_channels=64, out_channels=64, stride=(1, 1)),
            SpikingBTNK2(channels=256),
            SpikingBTNK2(channels=256)
        )

        self.stage_2 = nn.Sequential(
            SpikingBTNK1(in_channels=256, out_channels=128, stride=(2, 2)),
            SpikingBTNK2(channels=512),
            SpikingBTNK2(channels=512),
            SpikingBTNK2(channels=512)
        )

        self.stage_3 = nn.Sequential(
            SpikingBTNK1(in_channels=512, out_channels=256, stride=(2, 2)),
            SpikingBTNK2(channels=1024),
            SpikingBTNK2(channels=1024),
            SpikingBTNK2(channels=1024),
            SpikingBTNK2(channels=1024),
            SpikingBTNK2(channels=1024)
        )

        self.stage_4 = nn.Sequential(
            SpikingBTNK1(in_channels=1024, out_channels=512, stride=(2, 2)),
            SpikingBTNK2(channels=2048),
            SpikingBTNK2(channels=2048)
        )
        self.num_channels = 2048

    def forward(self, x: NestedTensor):
        out: Dict[str, NestedTensor] = {}
        xs = x.tensors
        m = x.mask
        assert m is not None
        T = xs.shape[0]
        mask_ = []
        xs = self.stage_0(xs)
        xs = self.stage_1(xs)
        for i in range(T):
            mask_.append(F.interpolate(m[0][None].float(), size=xs.shape[-2:]).to(torch.bool)[0])
        out["layer1"] = NestedTensor(xs, mask_)
        mask_.clear()
        xs = self.stage_2(xs)
        for i in range(T):
            mask_.append(F.interpolate(m[0][None].float(), size=xs.shape[-2:]).to(torch.bool)[0])
        out["layer2"] = NestedTensor(xs, mask_)
        mask_.clear()
        xs = self.stage_3(xs)
        for i in range(T):
            mask_.append(F.interpolate(m[0][None].float(), size=xs.shape[-2:]).to(torch.bool)[0])
        out["layer3"] = NestedTensor(xs, mask_)
        mask_.clear()
        xs = self.stage_4(xs)
        for i in range(T):
            mask_.append(F.interpolate(m[0][None].float(), size=xs.shape[-2:]).to(torch.bool)[0])
        out["layer4"] = NestedTensor(xs, mask_)
        mask_.clear()
        return out


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        # 经过backbone
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        # 每个特征层应该是，还是每个image
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))
        return out, pos


def build_backbone(args):
    # 创建位置编码
    position_embedding = build_position_encoding(args)
    backbone = NewSpikingResNet()
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
