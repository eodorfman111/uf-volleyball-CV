import torch
import torch.nn as nn
import torch.nn.functional as F


def block(i, o):
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True), nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True)
    )


class HeatNet(nn.Module):
    def __init__(self, w=16, in_ch=3):
        super().__init__()
        self.e1, self.e2, self.e3 = block(in_ch, w), block(w, 2 * w), block(2 * w, 4 * w)
        self.d2, self.d1 = block(6 * w, 2 * w), block(3 * w, w)
        self.head = nn.Conv2d(w, 1, 1)

    def forward(self, x):
        a = self.e1(x)
        b = self.e2(F.max_pool2d(a, 2))
        c = self.e3(F.max_pool2d(b, 2))
        b = self.d2(torch.cat([F.interpolate(c, scale_factor=2), b], 1))
        a = self.d1(torch.cat([F.interpolate(b, scale_factor=2), a], 1))
        return self.head(a)


def load_weights(net, state, center=1):
    own = net.state_dict()
    with torch.no_grad():
        for k, v in state.items():
            if k not in own:
                continue
            t = own[k]
            if v.shape == t.shape:
                t.copy_(v)
            elif k == "e1.0.weight" and v.shape[1] == 3 and t.shape[1] % 3 == 0 and center * 3 + 3 <= t.shape[1]:
                t.zero_()
                t[:, center * 3 : center * 3 + 3] = v
            else:
                raise ValueError(f"cannot load {k}: {tuple(v.shape)} into {tuple(t.shape)}")
    net.load_state_dict(own)
