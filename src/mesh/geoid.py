# pyright: reportMissingImports=false
from collections import OrderedDict

import torch 
import torch.nn as nn

import trimesh

from src.mesh.faces import make_cube_faces


class Geoid(nn.Module):
    def __init__(self, n):
        super(Geoid, self).__init__()        
        stacked = self.make_cube_sides(n, stacked=True)         
        x_angle, z_angle = self.get_xz_angles(stacked)        
        self.register_buffer('x_angle', x_angle)
        self.register_buffer('z_angle', z_angle)
        self.register_buffer('faces', make_cube_faces(n).int())        
        self.register_buffer('colors', torch.ones(torch.numel(x_angle), 3) * 0.5)
        
        self.radii = torch.nn.Parameter(torch.zeros(3, *x_angle.shape))

    def to_vertices(_, t):
        return t.permute(0, 2, 3, 1).reshape(-1, 3)
    
    def get_xz_angles(_, bxyz):
        x, y, z = bxyz[:, 0, :, :], bxyz[:, 1, :, :], bxyz[:, 2, :, :]
        r = torch.sqrt(x**2 + y**2 + z**2)
        x_angle = torch.atan2(y, x)
        z_angle = torch.acos(z/r)
        return x_angle, z_angle
   
    def get_ellipsoidal(_, x_angle, z_angle, radii):
        x = radii[0] * torch.sin(z_angle) * torch.cos(x_angle)
        y = radii[1] * torch.sin(z_angle) * torch.sin(x_angle)
        z = radii[2] * torch.cos(z_angle) 
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
    
    
    def forward(self, radii=None):
        radii = torch.sigmoid(radii or self.radii)
        ellipsoidal = self.get_ellipsoidal(self.x_angle, self.z_angle, radii)        
        vert = self.to_vertices(ellipsoidal)
        return vert, self.faces

    def export(self, f, radii=None):        
        verts, faces, _ =  self.forward(radii)
        vertices = verts.cpu().detach()
        faces = faces.cpu().detach()
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.export(f)