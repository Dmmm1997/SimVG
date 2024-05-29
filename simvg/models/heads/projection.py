import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_layer(in_dim, out_dim, kernel_size=1, padding=0, stride=1):
    return nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=False), nn.BatchNorm2d(out_dim), nn.ReLU(True))


def linear_layer(in_dim, out_dim, bias=False):
    return nn.Sequential(nn.Linear(in_dim, out_dim, bias), nn.BatchNorm1d(out_dim), nn.ReLU(True))


class Projector(nn.Module):
    def __init__(self, word_dim=768, in_dim=768, kernel_size=3):
        super().__init__()
        self.in_dim = in_dim
        self.kernel_size = kernel_size
        # visual projector
        self.vis = nn.Conv2d(in_dim, in_dim // 2, 1)
        # textual projector
        out_dim = 2 * in_dim // 2 * kernel_size * kernel_size + 2
        self.txt = nn.Linear(word_dim, out_dim)

    def forward(self, x, word):
        """
        x: b, 512, 26, 26
        word: b, N, 512
        """
        x = self.vis(x)
        B, C, H, W = x.size()
        # 1, b*256, 104, 104
        x = x.reshape(1, B * C, H, W)
        # txt: b, (256*3*3 + 1) -> b, 256, 3, 3 / b
        word = self.txt(word)
        weight, bias = word[:, :-2], word[:, -2:]
        bias = bias.reshape(-1)
        weight = weight.reshape(B * 2, C, self.kernel_size, self.kernel_size)
        # Conv2d - 1, b*256, 104, 104 -> 1, b, 104, 104
        out = F.conv2d(x, weight, padding=self.kernel_size // 2, groups=B, bias=bias)
        out = out.transpose(0, 1)
        out = out.reshape(B, 2, H, W)
        # b, 1, 104, 104
        return out
