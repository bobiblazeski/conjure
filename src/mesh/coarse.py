# pyright: reportMissingImports=false
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.shared.gaussian import Gaussian
from src.shared.faces import make_cube_faces
from src.shared.sides import (    
    to_vertices,
    make_phi_theta,
    sphered_vertices,
    to_spherical,
)

class Coarse(nn.Module):
    def __init__(self, ns, kernel=3, sigma=1, r=0.5, ch=32, bias=True):
        super(Coarse, self).__init__()
        n = ns[-1]
        self.n = n
        phi, theta = make_phi_theta(n)
        self.register_buffer('phi', phi)
        self.register_buffer('theta', theta)
        self.register_buffer('faces', make_cube_faces(n))
        
        self.abc = nn.Parameter(torch.zeros(1, 3, 1, 1))
        self.radii = nn.ParameterList([
            nn.Parameter(torch.zeros(6, 3, l, l)) for l in ns])        
        

        self.gaussian =  Gaussian(kernel, sigma=sigma)
   
    # def get_ellipsoidal(self, radii, abc):
    #     x = (abc[0] + radii[:, 0]) * torch.sin(self.theta) * torch.cos(self.phi)
    #     y = (abc[1] + radii[:, 1]) * torch.sin(self.theta) * torch.sin(self.phi)
    #     z = (abc[2] + radii[:, 2]) * torch.cos(self.theta)
    #     return torch.stack((x, y, z), dim=1)   
    
    def get_ellipsoidal(self, radii):
        x = radii[:, 0] * torch.sin(self.theta) * torch.cos(self.phi)
        y = radii[:, 1] * torch.sin(self.theta) * torch.sin(self.phi)
        z = radii[:, 2] * torch.cos(self.theta)
        return torch.stack((x, y, z), dim=1) 
    
    def scale(_, t, sz):
        return F.interpolate(t, sz, mode='bilinear', align_corners=True)
    
    def forward(self): 
        radii = self.gaussian(self.radii[0])
        for high in self.radii[1:]:            
            radii = self.scale(radii, high.size(-1)) + self.gaussian(high)
            
        radii = torch.sigmoid(radii +self.abc)
        #abc = torch.sigmoid(radii + self.abc) 
        stacked = self.get_ellipsoidal(radii)
        vert = to_vertices(stacked)
        return vert, self.faces, radii, stacked

    def save(self, filename):
        torch.save(self.radii.data, filename)

    def load(self, filename):
        self.radii = nn.Parameter(torch.load(filename))