import torch
import torch.nn as nn
import torch.nn.functional as F


def block(i, o):
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True), nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True)
    )


class HeatNet(nn.Module):
    def __init__(self, w=16):
        super().__init__()
        self.e1, self.e2, self.e3 = block(3, w), block(w, 2 * w), block(2 * w, 4 * w)
        self.d2, self.d1 = block(6 * w, 2 * w), block(3 * w, w)
        self.head = nn.Conv2d(w, 1, 1)

    def forward(self, x):
        a = self.e1(x)
        b = self.e2(F.max_pool2d(a, 2))
        c = self.e3(F.max_pool2d(b, 2))
        b = self.d2(torch.cat([F.interpolate(c, scale_factor=2), b], 1))
        a = self.d1(torch.cat([F.interpolate(b, scale_factor=2), a], 1))
        return self.head(a)
