
# pyright: reportMissingImports=false
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

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

def sphered_vertices(n, r=0.5):
     start, end= -r, +r
     d1, d2 = torch.meshgrid(
         torch.linspace(start, end, steps=n),
         torch.linspace(start, end, steps=n))
     d3 = torch.full_like(d1, end) + 1 / n
     sides =  OrderedDict({
         'front': torch.stack((+d3,  d1,  d2), dim=0),
         'right': torch.stack(( d1, +d3,  d2), dim=0),    
         'back' : torch.stack((-d3,  d1,  d2), dim=0),         
         'left' : torch.stack(( d1, -d3,  d2), dim=0),
         'top'  : torch.stack(( d1,  d2, +d3), dim=0),
         'down' : torch.stack(( d1,  d2, -d3), dim=0),
     })
     stacked = torch.stack([p for p in sides.values()])
     sphered = cube_to_sphere(stacked)
     return sphered.permute(0, 2, 3, 1).reshape(-1, 3)
      