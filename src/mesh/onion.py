# pyright: reportMissingImports=false

import torch
import torch.nn as nn

from src.mesh.ellipsoid import Ellipsoid
from src.mesh.offset_normals import OffsetNormals

from src.shared.sides import (    
    to_vertices,    
)


class Onion(nn.Module):
    def __init__(self, ns, ratio=0.0125):        
        super().__init__()    
        self.radii = torch.nn.Parameter(torch.Tensor([.5, .5, .5]))
        self.core =  Ellipsoid(ns[0])
        self.layers = [OffsetNormals(n, ratio=ratio) for n in ns[1:]]
        
    def forward(self):
        bxyz, _ = self.core(self.radii)
        for layer in self.layers:
            bxyz, f = layer(bxyz)        
        return to_vertices(bxyz), f