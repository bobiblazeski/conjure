# pyright: reportMissingImports=false
import torch
import torch.nn as nn

class EdgeLoss(nn.Module):
    def __init__(self, l1=True):
        super(EdgeLoss, self).__init__()        
        self.loss = nn.L1Loss() if l1 else nn.MSELoss()

    def forward(self, x):
        losses = [
            self.loss(x[0, :,  :,  0], x[5, :, -1,  :]), 
            self.loss(x[0, :,  :, -1], x[4, :, -1,  :]),
            self.loss(x[0, :,  0,  :], x[3, :, -1,  :]),
            self.loss(x[0, :, -1,  :], x[1, :, -1,  :]),
            self.loss(x[1, :,  :,  0], x[5, :,  :, -1]),
            self.loss(x[1, :,  :, -1], x[4, :,  :, -1]),
            self.loss(x[1, :,  0,  :], x[2, :, -1,  :]),
            self.loss(x[2, :,  :,  0], x[5, :,  0,  :]),
            self.loss(x[2, :,  :, -1], x[4, :,  0,  :]),
            self.loss(x[2, :,  0,  :], x[3, :,  0,  :]),
            self.loss(x[3, :,  :,  0], x[5, :,  :,  0]),
            self.loss(x[3, :,  :, -1], x[4, :,  :,  0]),
        ]
        return sum(losses) / len(losses)