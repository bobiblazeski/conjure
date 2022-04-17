
# pyright: reportMissingImports=false
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.shared.faces import make_cube_faces
from src.shared.sides import (
    avg_border,
    to_vertices,
    to_stacked,
)

from scripts.geometry import (
    compute_vertex_normals, 
    compute_face_normals,
)

class OffsetNormals(nn.Module):
    def __init__(self, n, ratio=0.0125):        
        super().__init__()
        self.ratio = ratio
        self.n = n
        self.ps = nn.Parameter(torch.randn((6, 1, n, n)))
        self.register_buffer('faces', make_cube_faces(n).int())  
        
    def scale(self, t):
        return F.interpolate(t, self.n, mode='bilinear', align_corners=True)


    def forward(self, bxyz):
        vrt = self.scale(bxyz)
        #vrt = avg_border(vrt) # Makes nan
        vrt = to_vertices(vrt)

        qs =  torch.sigmoid(self.ps)        
        face_normals = compute_face_normals(vrt, self.faces)
        vrt_normals = compute_vertex_normals(vrt, self.faces, face_normals)
        offset = to_stacked(vrt_normals) * qs * self.ratio             
        return to_stacked(vrt) + offset, self.faces