# pyright: reportMissingImports=false
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.linalg import vector_norm

import pytorch3d 
import pytorch3d.ops
from pytorch3d.ops.knn import knn_points

from src.shared.faces import make_cube_faces

class Layer(nn.Module):
    def __init__(self, n, coarse, ratio=0.5, align_corners=True):
        super(Layer, self).__init__()
        self.ratio = ratio
        start = F.interpolate(coarse, n, mode='bilinear', align_corners=align_corners)
        self.offsets = nn.Parameter(torch.randn(6, 3, n, n))
        self.register_buffer('start', start)
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
    
    def forward(self):
        offsets = torch.tanh(self.offsets)
        normed = torch.nan_to_num(offsets / vector_norm(offsets, dim=1, keepdim=True))
        vertices = self.start + normed * self.scales
        return vertices