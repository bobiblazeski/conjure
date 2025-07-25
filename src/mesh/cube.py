# pyright: reportMissingImports=false
from collections import OrderedDict

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.shared.gaussian import Gaussian
from src.shared.faces import make_cube_faces
from src.shared.padding import pad_side
from src.shared.singan import Generator
from src.shared.sides import (
    sphered_vertices,
    to_vertices,
)

def sides_dict(n):
    return nn.ParameterDict({
        'front': nn.Parameter(torch.zeros((1, 3, n, n))),
        'back' : nn.Parameter(torch.zeros((1, 3, n, n))),
        'left' : nn.Parameter(torch.zeros((1, 3, n, n))),
        'right': nn.Parameter(torch.zeros((1, 3, n, n))),
        'top'  : nn.Parameter(torch.zeros((1, 3, n, n))),
        'down' : nn.Parameter(torch.zeros((1, 3, n, n))),
    })

def cube_to_sphere(bxyz):
    x, y, z = bxyz[:, 0, :, :], bxyz[:, 1, :, :], bxyz[:, 2, :, :]
    r = torch.sqrt(x**2 + y**2 + z**2)
    x_angle = torch.atan2(y, x)
    z_angle = torch.acos(z/r)
    
    r = torch.ones_like(x) * 0.5
    x = r * torch.sin(z_angle) * torch.cos(x_angle)
    y = r * torch.sin(z_angle) * torch.sin(x_angle)
    z = r * torch.cos(z_angle)        
    return torch.stack((x, y, z), dim=1)   

def dict_to_stack(params):
    return torch.cat([p for p in params.values()])

def stack_to_dict(stacked):
    return OrderedDict({
        'front': stacked[0].permute(1, 2, 0),
        'right': stacked[1].permute(1, 2, 0),    
        'back' : stacked[2].permute(1, 2, 0),         
        'left' : stacked[3].permute(1, 2, 0),
        'top'  : stacked[4].permute(1, 2, 0),
        'down' : stacked[5].permute(1, 2, 0),
    })

class SimpleCube(nn.Module):
    def __init__(self, n, padding=False, kernel=5, sigma=1, r=0.5):
        super(SimpleCube, self).__init__()        
        self.n = n
        self.padding = padding
        self.kernel = kernel
        self.register_buffer('start', sphered_vertices(n, r))
        self.register_buffer('faces', make_cube_faces(n).int())
        self.params = sides_dict(n)
        self.gaussian = Gaussian(kernel, sigma=sigma, padding=not padding)
        
    def pad(self, stacked):
        sides = stack_to_dict(stacked)
        res = OrderedDict()
        for side_name in sides.keys():
            padded = pad_side(sides, side_name, self.kernel)
            res[side_name] = padded.permute(2, 0, 1)[None]
        return dict_to_stack(res)
    
    def forward(self):
        vert = dict_to_stack(self.params)         
        if self.padding:
            vert = self.pad(vert)                            
        vert = self.gaussian(vert) + self.start        
        return to_vertices(vert), self.faces

class ProgressiveCube(nn.Module):
    def __init__(self, n, ns, padding=False, kernel=5, sigma=1, r=0.5):
        super(ProgressiveCube, self).__init__()
        self.n = n
        self.ns = ns
        self.padding = padding
        self.kernel = kernel
        self.register_buffer('start', sphered_vertices(n, r))
        self.register_buffer('faces', make_cube_faces(n).int())
        self.params = nn.ModuleList([sides_dict(n) for n in ns])
        self.gaussian = Gaussian(kernel, sigma=sigma, padding=not padding)
    
    def scale(self, t):
        return  F.interpolate(t, self.n, mode='bilinear', align_corners=True)
    
    def pad(self, stacked):
        sides = stack_to_dict(stacked)
        res = OrderedDict()
        for side_name in sides.keys():
            padded = pad_side(sides, side_name, self.kernel)
            res[side_name] = padded.permute(2, 0, 1)[None]
        return dict_to_stack(res)

    def forward(self):
        res = self.start
        for params in self.params:
            vert = self.scale(dict_to_stack(params))
            if self.padding:
                vert = self.pad(vert)                            
            res = res + self.gaussian(vert)
        return to_vertices(res), self.faces

class NetCube(nn.Module):
    def __init__(self, n, opt, kernel=5, sigma=1, r=0.5):
        super(NetCube, self).__init__()        
        self.n = n
        self.r = r
        self.kernel = kernel
        self.register_buffer('start', sphered_vertices(n + 2 * opt.num_layer, r))
        self.register_buffer('faces', make_cube_faces(n).int())        
        self.gaussian = Gaussian(kernel, sigma=sigma, padding=True)
        self.net = Generator(opt)

    def get_start(self):
        return to_vertices(sphered_vertices(self.n, self.r))
    
    def forward(self):
        vert = self.net(self.start)        
        vert = self.gaussian(vert) 
        return to_vertices(vert), self.faces