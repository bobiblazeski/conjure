# pyright: reportMissingImports=false
from collections import OrderedDict

import torch
import torch.nn as nn

from src.shared.faces import make_cube_faces
from src.shared.sides import (    
    to_vertices,
    make_phi_theta,
    sphered_vertices,
    to_spherical,
)

class Coarse(nn.Module):
    def __init__(self, n, r=0.5, ch=32):
        super(Coarse, self).__init__()        
        self.n = n
        phi, theta = make_phi_theta(n)
        self.register_buffer('phi', phi)
        self.register_buffer('theta', theta)
        self.register_buffer('faces', make_cube_faces(n))                
        self.register_buffer('sphere1', to_spherical(sphered_vertices(n + (2*3), r)))
        self.register_buffer('sphere2', sphered_vertices(n, r))
        self.net = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, ch, 3, bias=False)),
            ('relu1', nn.LeakyReLU(0.1)),            
            ('conv2', nn.Conv2d(ch, ch, 3, bias=False)),
            ('relu2', nn.LeakyReLU(0.1)),            
            ('conv3', nn.Conv2d(ch, 3, 3, bias=False)),
            ('act3', torch.nn.Sigmoid()),
        ]))
        self.radii = torch.nn.Parameter(torch.zeros(6, 3, n, n))
   
    def get_ellipsoidal(self, radii):
        x = radii[:, 0] * torch.sin(self.theta) * torch.cos(self.phi)
        y = radii[:, 1] * torch.sin(self.theta) * torch.sin(self.phi)
        z = radii[:, 2] * torch.cos(self.theta)
        return torch.stack((x, y, z), dim=1)   
    
    def forward(self):        
        #radii = self.net(self.sphere1 + torch.randn_like(self.sphere1) * 0.05) / 2 + 0.5
        ellipsoidal = self.get_ellipsoidal(torch.sigmoid(self.radii))
        vert = to_vertices(ellipsoidal)
        return vert, self.faces, self.radii