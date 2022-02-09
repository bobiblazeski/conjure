
# pyright: reportMissingImports=false
from collections import OrderedDict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def to_vertices(stacked):
    return stacked.permute(0, 2, 3, 1).reshape(-1, 3)

def to_stacked(vs):
    n = int(math.sqrt(vs.size(0) // 6))    
    return vs.reshape(6, n, n, 3).permute(0, 3, 1, 2)

def sides_dict(n):
    return nn.ParameterDict({
        'front': nn.Parameter(torch.zeros((1, 3, n, n))),
        'back' : nn.Parameter(torch.zeros((1, 3, n, n))),
        'left' : nn.Parameter(torch.zeros((1, 3, n, n))),
        'right': nn.Parameter(torch.zeros((1, 3, n, n))),
        'top'  : nn.Parameter(torch.zeros((1, 3, n, n))),
        'down' : nn.Parameter(torch.zeros((1, 3, n, n))),
    })

def cube_to_sphere(bxyz, r_sphere):
    x, y, z = bxyz[:, 0, :, :], bxyz[:, 1, :, :], bxyz[:, 2, :, :]
    r = torch.sqrt(x**2 + y**2 + z**2)
    x_angle = torch.atan2(y, x)
    z_angle = torch.acos(z/r)
        
    x = r_sphere * torch.sin(z_angle) * torch.cos(x_angle)
    y = r_sphere * torch.sin(z_angle) * torch.sin(x_angle)
    z = r_sphere * torch.cos(z_angle)        
    return torch.stack((x, y, z), dim=1)   

def cubed_vertices(n, r):
    start, end= -r, +r
    d1, d2 = torch.meshgrid(
         torch.linspace(start, end, steps=n),
         torch.linspace(start, end, steps=n),
         indexing='ij')
    d3 = torch.full_like(d1, end) + 1 / n
    sides =  OrderedDict({
        'front': torch.stack((+d3,  d1,  d2), dim=0),
        'right': torch.stack(( d1, +d3,  d2), dim=0),    
        'back' : torch.stack((-d3,  d1,  d2), dim=0),         
        'left' : torch.stack(( d1, -d3,  d2), dim=0),
        'top'  : torch.stack(( d1,  d2, +d3), dim=0),
        'down' : torch.stack(( d1,  d2, -d3), dim=0),
    })  
    return torch.stack([p for p in sides.values()])
     

def sphered_vertices(n, r):
    stacked = cubed_vertices(n, r)
    return cube_to_sphere(stacked, r)

def make_phi_theta(n):
    bxyz = sphered_vertices(n, 1)
    x, y, z = bxyz[:, 0, :, :], bxyz[:, 1, :, :], bxyz[:, 2, :, :]
    r = torch.sqrt(x**2 + y**2 + z**2)
    phi = torch.atan2(y, x)
    theta = torch.acos(z/r)
    return phi, theta

def to_spherical(bxyz):
    x, y, z = bxyz[:, 0, :, :], bxyz[:, 1, :, :], bxyz[:, 2, :, :]
    radii = torch.sqrt(x**2 + y**2 + z**2)
    phi = torch.atan2(y, x)
    theta = torch.acos(z/radii)
    return torch.stack((phi, theta, radii), dim=1)

def set_edges(sides, edges):
    x = sides + 0
    x[0, :,  :,  0] = edges[0]
    x[5, :, -1,  :] = edges[0]
    
    x[0, :,  :, -1] = edges[1]
    x[4, :, -1,  :] = edges[1]

    x[0, :,  0,  :] = edges[2]
    x[3, :, -1,  :] = edges[2]

    x[0, :, -1,  :] = edges[3]
    x[1, :, -1,  :] = edges[3]

    x[1, :,  :,  0] = edges[4]
    x[5, :,  :, -1] = edges[4]

    x[1, :,  :, -1] = edges[5]
    x[4, :,  :, -1] = edges[5]

    x[1, :,  0,  :] = edges[6]
    x[2, :, -1,  :] = edges[6]

    x[2, :,  :,  0] = edges[7]
    x[5, :,  0,  :] = edges[7]

    x[2, :,  :, -1] = edges[8]
    x[4, :,  0,  :] = edges[8]

    x[2, :,  0,  :] = edges[9]
    x[3, :,  0,  :] = edges[9]

    x[3, :,  :,  0] = edges[10]
    x[5, :,  :,  0] = edges[10]

    x[3, :,  :, -1] = edges[11]
    x[4, :,  :,  0] = edges[11]
    return x

def set_corners(sides, corners):
    x = sides + 0
    x[0, :,  0, 0] = corners[0]
    x[3, :, -1, 0] = corners[0]
    x[5, :, -1, 0] = corners[0]
    
    x[0, :,  0, -1] = corners[1]
    x[3, :, -1, -1] = corners[1]
    x[4, :, -1,  0] = corners[1]
    
    x[0, :, -1,  0] = corners[2]
    x[1, :, -1,  0] = corners[2]
    x[5, :, -1, -1] = corners[2]
    
    x[0, :, -1, -1] = corners[3]
    x[1, :, -1, -1] = corners[3]
    x[4, :, -1, -1] = corners[3]
    
    x[1, :,  0,  0] = corners[4]
    x[2, :, -1,  0] = corners[4]
    x[5, :,  0, -1] = corners[4]
    
    x[1, :,  0, -1] = corners[5]
    x[2, :, -1, -1] = corners[5]
    x[4, :,  0, -1] = corners[5]
    
    x[2, :,  0,  0] = corners[6]
    x[3, :,  0,  0] = corners[6]
    x[5, :,  0,  0] = corners[6]
    
    x[2, :,  0, -1] = corners[7]
    x[3, :,  0, -1] = corners[7]
    x[4, :,  0,  0] = corners[7]
    return x    