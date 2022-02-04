# pyright: reportMissingImports=false
from collections import OrderedDict

import torch
import torch.nn as nn

class LaplacianLoss(nn.Module):
    def __init__(self, channels=3, padd=False):
        super(LaplacianLoss, self).__init__()
        arr = [('padd', nn.ReflectionPad2d(2))] if padd else []
        arr.append(('conv', nn.Conv2d(channels, channels, 3, 
            stride=1, padding=0, bias=False, groups=channels)),)
        self.seq = nn.Sequential(OrderedDict(arr))
        self.seq.requires_grad_(False)
        self.weights_init()

    def forward(self, x):
        return self.seq(x).abs().mean()

    def weights_init(self):
        w = torch.tensor([[ 1.,   4., 1.],
                          [ 4., -20., 4.],
                          [ 1.,   4., 1.],])
        for _, f in self.named_parameters():           
            f.data.copy_(w)
