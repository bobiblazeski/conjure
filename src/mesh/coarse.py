# pyright: reportMissingImports=false
from collections import OrderedDict

import torch
import torch.nn as nn

from src.shared.gaussian import Gaussian
from src.shared.faces import make_cube_faces
from src.shared.sides import (    
    to_vertices,
    make_phi_theta,
    sphered_vertices,
    to_spherical,
)

class Coarse(nn.Module):
    def __init__(self, n, kernel=7, sigma=1, r=0.5, ch=32, bias=True):
        super(Coarse, self).__init__()        
        self.n = n
        phi, theta = make_phi_theta(n)
        self.register_buffer('phi', phi)
        self.register_buffer('theta', theta)
        self.register_buffer('faces', make_cube_faces(n))
        
        nx =  n + (2*3)
        sphere1  = to_spherical(sphered_vertices(nx, r))
        bc = torch.zeros(6, 2, nx, nx) + r
        sphere1 = torch.cat((sphere1, bc), dim=1)
        self.register_buffer('sphere1', sphere1)
        self.register_buffer('sphere2', sphered_vertices(n, r))
        
        self.net = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(5, ch, 3, bias=bias)),
            ('relu1', torch.nn.ReLU()), # nn.LeakyReLU(0.1)
            ('conv2', nn.Conv2d(ch, ch, 3, bias=bias)),
            ('relu2', torch.nn.ReLU()), # nn.LeakyReLU(0.1)
            ('conv3', nn.Conv2d(ch, 3, 3, bias=bias)),
            ('act3', torch.nn.Sigmoid()),
        ]))
        self.abc = nn.Parameter(torch.zeros(3))
        #self.radii = torch.nn.Parameter(torch.zeros(6, 3, n, n))
        #self.abc = nn.Parameter(torch.randn(3))
        #self.radii = torch.nn.Parameter(torch.randn(6, 3, n, n))
        #self.abc = nn.Parameter(torch.rand(3)-0.5)
        #self.radii = torch.nn.Parameter(torch.rand(6, 3, n, n))

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
    
    def forward(self):        
        radii = self.net(self.sphere1) + 0.5
        #radii = torch.sigmoid(self.gaussian(self.radii))
        #abc = torch.sigmoid(radii + self.abc) 
        stacked = self.get_ellipsoidal(radii)
        vert = to_vertices(stacked)
        return vert, self.faces, radii, stacked

    def save(self, filename):
        torch.save(self.radii.data, filename)

    def load(self, filename):
        self.radii = nn.Parameter(torch.load(filename))
    