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

class Octave(nn.Module):
    def __init__(self, n, factor=1,  noise=0.025):
        super(Octave, self).__init__()        
        self.n = n        
        self.factor = 2**-factor
        self.register_buffer('faces', make_cube_faces(n))        
        self.ps = nn.Parameter(torch.zeros(6, 1, n, n) * noise)        
    
    def scale(self, t):
        return  F.interpolate(t, self.n, mode='bilinear', align_corners=True)
            
    def get_vertex_normals(self, vert):
        face_normals = compute_face_normals(vert, self.faces)
        normals = compute_vertex_normals(vert, self.faces, face_normals)
        normals = to_stacked(normals)
        normals[0] *= -1 
        normals[3] *= -1 
        normals[4] *= -1        
        return to_vertices(normals)

    def forward(self, vrt, nrm):
        vrt, nrm = [to_stacked(f) for f in [vrt, nrm]]
        vrt, nrm = self.scale(vrt), self.scale(nrm).norm(dim=1, keepdim=True)
        stacked = vrt + self.factor * torch.sigmoid(self.ps) * nrm
        
        vert = to_vertices(stacked)                
        normals = self.get_vertex_normals(vert)
        return vert, self.faces, normals