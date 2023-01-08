# pyright: reportMissingImports=false
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.linalg import vector_norm

import pytorch3d 
import pytorch3d.ops
from pytorch3d.ops.knn import knn_points

from src.shared.faces import make_cube_faces
from src.shared.sides import (
    sphered_vertices,
    to_vertices,
    to_stacked,
)

class Core(nn.Module):
    def __init__(self, n, ratio=0.5):
        super(Core, self).__init__()        
        self.ratio = ratio
        self.center = nn.Parameter(torch.zeros(1, 3, 1, 1))
        self.offsets = nn.Parameter(torch.randn(6, 3, n, n) )
        self.register_buffer('start', sphered_vertices(n, 1))
        self.register_buffer('faces', make_cube_faces(n).int())
        self.register_buffer('scales', self.get_scales(self.start, self.ratio))
        
    def get_scales(self, vertices, ratio):
        with torch.no_grad():            
            n = vertices.size(-1)
            pts = vertices.permute(0, 2, 3, 1).reshape(-1, 3)[None]
            dists = knn_points(pts, pts, K=2).dists[:, :, 1]
            dists = dists.reshape(6, 1, n, n)            
            return dists * ratio
            
        
    def rescale(self):
        self.scales.data.set_(self.get_scales(self.forward(), self.ratio))         
        # What about minimum and maximum distances
        # to keep surface smooth
        # That should be decided by the ratio??
    
    
    def forward(self):
        offsets = torch.tanh(self.offsets)
        normed = torch.nan_to_num(offsets / vector_norm(offsets, dim=1, keepdim=True))
        vertices = self.center + self.start + normed * self.scales
        return vertices

