# pyright: reportMissingImports=false
import numpy  as np
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter

class Gaussian(nn.Module):
    def __init__(self, kernel_size=3, sigma=1, padding=1,  channels=3):
        super(Gaussian, self).__init__()        
        assert kernel_size % 2 == 1
        self.kernel_size = kernel_size
        self.side = (kernel_size-1) // 2
        self.sigma = sigma
        self.padding = padding
        self.seq = nn.Sequential(
            nn.ReplicationPad2d(padding), 
            nn.Conv2d(channels, channels, self.kernel_size, stride=1, padding=0, bias=None, groups=channels)
        )
        self.seq.requires_grad_(False)
        self.weights_init()

    def forward(self, x):
        return self.seq(x)

    def weights_init(self):
        n= np.zeros((self.kernel_size, self.kernel_size))
        n[self.side, self.side] = 1
        k = gaussian_filter(n, sigma=self.sigma)
        for _, f in self.named_parameters():            
            f.data.copy_(torch.from_numpy(k))
