# pyright: reportMissingImports=false
import torch
import torch.nn as nn

from src.shared.sides import (    
    avg_border,
    to_vertices,
    make_phi_theta,
)

from scripts.geometry import (
    compute_vertex_normals, 
    compute_face_normals,
)
from src.shared.sides import (    
    avg_border,
    to_vertices,
    to_stacked,
)
from src.shared.faces import make_cube_faces


class Core(nn.Module):
    def __init__(self, n, noise=0.025, data_dir='./data/', load=True):
        super(Core, self).__init__()        
        self.n = n
        phi, theta = make_phi_theta(n)        
        self.register_buffer('angles', torch.stack((
            torch.sin(theta) * torch.cos(phi),
            torch.sin(theta) * torch.sin(phi),
            torch.cos(theta),
        ), dim=1))
        self.register_buffer('faces', make_cube_faces(n))        
        self.radii = nn.Parameter(torch.randn(6, 1, n, n) * noise)        
    
    def get_vertex_normals(self, vert):
        face_normals = compute_face_normals(vert, self.faces)
        normals = compute_vertex_normals(vert, self.faces, face_normals)
        normals = to_stacked(normals)
        normals[0] *= -1 
        normals[3] *= -1 
        normals[4] *= -1        
        return to_vertices(normals)

    def forward(self):         
        stacked = torch.sigmoid(avg_border(self.radii)) * self.angles
        vert = to_vertices(stacked)                
        normals = self.get_vertex_normals(vert)
        return vert, self.faces, normals
