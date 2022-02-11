# pyright: reportMissingImports=false
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.shared.gaussian import Gaussian
from src.shared.faces import make_cube_faces
from src.shared.sides import (    
    to_vertices,
    make_phi_theta,
    sphered_vertices,
    to_spherical,
    set_edges,
    set_corners,
)

import torch
from pytorch_wavelets import DWTForward, DWTInverse

# torch.Size([6, 3, 8, 8])
# torch.Size([6, 3, 3, 18, 18])
# torch.Size([6, 3, 3, 11, 11])
# torch.Size([6, 3, 3, 8, 8])

class WaveGenerator(nn.Module):
    def __init__(self, n, kernel=3, sigma=1, padding=1, r=0.5, scale=0.01):
        super(WaveGenerator, self).__init__()        
        self.n = 32
        self.scale = scale
        self.register_buffer('faces', make_cube_faces(n))        
        self.register_buffer('coarse', sphered_vertices(8, r))
        
        self.lvl1 = nn.Parameter(torch.randn(6, 3, 3, 18, 18) * scale)        
        self.lvl2 = nn.Parameter(torch.randn(6, 3, 3, 11, 11) * scale) 
        self.lvl3 = nn.Parameter(torch.randn(6, 3, 3, 8, 8) * scale) 

        self.gaussian =  Gaussian(kernel, sigma=sigma, padding=padding)
        
        self.wave = 'db3'
        self.mode = 'symmetric'
        self.levels = 3
        self.xfm  = DWTForward(J=self.levels, mode=self.mode, wave=self.wave)
        self.ifm = DWTInverse(mode=self.mode, wave=self.wave) 
       
    def forward(self): 
        Yl = self.coarse
        Yh = [torch.tanh(lvl) *m for (lvl, m) in zip(
          [self.lvl1 , self.lvl2, self.lvl3],
          [0.01, 0.1, 1.]
        )]
        stacked = self.ifm((Yl, Yh))        
        vert = to_vertices(stacked)
        return vert, self.faces, stacked