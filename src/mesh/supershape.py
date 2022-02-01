# pyright: reportMissingImports=false
import meshplot

import torch 
import torch.nn as nn

import trimesh


from src.shared.faces import make_cube_faces
from src.shared.sides import (    
    to_vertices,
    make_phi_theta,
)

class Supershape(nn.Module):
    def __init__(self, n):
        super(Supershape, self).__init__()        
        phi, theta = make_phi_theta(n)        
        self.register_buffer('phi', phi)
        self.register_buffer('theta', theta)
        self.register_buffer('faces', make_cube_faces(n).int())        
        self.params = nn.ParameterDict({
            'm': nn.Parameter(torch.tensor(2.)),
            'a': nn.Parameter(torch.tensor(1.)),
            'b': nn.Parameter(torch.tensor(1.)),
            'n1': nn.Parameter(torch.tensor(1.)),
            'n2': nn.Parameter(torch.tensor(2.)),
            'n3': nn.Parameter(torch.tensor(2.)),
        })
    
    def get_radius(self, angle, m, a, b, n1, n2, n3):
        return torch.pow(
           torch.pow(torch.abs(torch.cos(m*angle/4.)/a), n2) + 
           torch.pow(torch.abs(torch.sin(m*angle/4.)/b), n3)
        ,-1 / n1)
        
        
    def get_superformula(self):        
        m, a, b, n1, n2, n3 = [self.params[p]
            for p in ['m', 'a', 'b', 'n1', 'n2', 'n3']]
        r1 = self.get_radius(self.phi, m, a, b, n1, n2, n3)
        r2 = self.get_radius(self.theta, m, a, b, n1, n2, n3)
        x = r1 * torch.sin(self.theta) * r2 *torch.cos(self.phi)
        y = r1 * torch.sin(self.theta) * r2 * torch.sin(self.phi)
        z = r2 * torch.cos(self.theta) 
        return torch.stack((x, y, z), dim=1)
    
    def forward(self):
        xyz = self.get_superformula()        
        vert = to_vertices(xyz)
        return vert, self.faces

    def export(self, file):        
        v, f =  self.forward()
        v, f = v.cpu().detach(), f.cpu().detach()
        mesh = trimesh.Trimesh(vertices=v, faces=f)
        mesh.export(file)
        
    def plot(self):
        v, f = self.forward()
        meshplot.plot(v.detach().numpy(), f.numpy())
