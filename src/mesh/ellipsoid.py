# pyright: reportMissingImports=false
from collections import OrderedDict

import torch 
import torch.nn as nn

import trimesh

from src.shared.faces import make_cube_faces


class Ellipsoid(nn.Module):
    def __init__(self, n):
        super(Ellipsoid, self).__init__()        
        stacked = self.make_cube_sides(n, stacked=True)         
        x_angle, z_angle = self.get_xz_angles(stacked)        
        self.register_buffer('x_angle', x_angle)
        self.register_buffer('z_angle', z_angle)
        self.register_buffer('faces', make_cube_faces(n).int())        
    
    def get_xz_angles(_, bxyz):
        x, y, z = bxyz[:, 0, :, :], bxyz[:, 1, :, :], bxyz[:, 2, :, :]
        r = torch.sqrt(x**2 + y**2 + z**2)
        x_angle = torch.atan2(y, x)
        z_angle = torch.acos(z/r)
        return x_angle, z_angle
   
    def get_ellipsoidal(_, x_angle, z_angle, rs):
        x = rs[0] * torch.sin(z_angle) * torch.cos(x_angle)
        y = rs[1] * torch.sin(z_angle) * torch.sin(x_angle)
        z = rs[2] * torch.cos(z_angle) 
        return torch.stack((x, y, z), dim=1)
    
    def make_cube_sides(_, n, r=0.5, stacked=False):
        start, end =  -r, +r
        d1, d2 = torch.meshgrid(
            torch.linspace(start, end, steps=n),
            torch.linspace(start, end, steps=n))
        d3 = torch.full_like(d1, end) + 1 / n
        sides = OrderedDict({
            'front': torch.stack((+d3,  d1,  d2), dim=0),
            'right': torch.stack(( d1, +d3,  d2), dim=0),    
            'back' : torch.stack((-d3,  d1,  d2), dim=0),         
            'left' : torch.stack(( d1, -d3,  d2), dim=0),
            'top'  : torch.stack(( d1,  d2, +d3), dim=0),
            'down' : torch.stack(( d1,  d2, -d3), dim=0),
        })
        if stacked:
            return torch.stack([p for p in sides.values()])
        return sides
    
    
    def forward(self, rs, stacked=True):
        ellipsoidal = self.get_ellipsoidal(self.x_angle, self.z_angle, rs)        
        vert = ellipsoidal if stacked else self.to_vertices(ellipsoidal)
        return vert, self.faces

    def export(self, f, rs):        
        verts, faces, _ =  self.forward(rs)
        vertices = verts.cpu().detach()
        faces = faces.cpu().detach()
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.export(f)