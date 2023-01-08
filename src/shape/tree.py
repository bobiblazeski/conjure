from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.linalg import vector_norm

import pytorch3d 
import pytorch3d.ops
from pytorch3d.ops.knn import knn_points

from src.shared.faces import make_cube_faces
from src.shared.sides import (
    sphered_vertices,
    to_vertices,
    to_stacked,
)

def conv_layer(n):
    conv = nn.Conv2d(3, 3, kernel_size=n, stride=n, bias=None, groups=3)
    conv.weight.data.copy_(torch.zeros_like(conv.weight) + 1/n**2)
    conv.requires_grad_(False)
    return conv

def mean_variance(pts, conv, n):    
    mean = conv(pts)
    while mean.size(-1) >  n:
        mean = conv(mean)        
    up = F.interpolate(mean, size=pts.size(-1), mode='nearest')
    return pts-up, mean

def treeify(pts, layers, scale_factor):
    center = pts.mean(dim=(0, 2, 3), keepdim=True)
    pts = pts - center    
    conv = conv_layer(scale_factor)
    offsets = [center]
    for layer in sorted(layers):        
        pts, mean = mean_variance(pts, conv, layer)
        offsets.append(mean)
    offsets.append(pts)
    return offsets  

def get_untils(offsets):
    until = offsets[0]
    untils = [until.clone()]
    for t in offsets[1:]:    
        until = t + F.interpolate(until, size=t.size(-1), mode='nearest')
        untils.append(until.clone())
    return untils

class Core(nn.Module):
    def __init__(self, start):
        super(Core, self).__init__()              
        self.offsets = nn.Parameter(torch.zeros_like(start))
        self.register_buffer('start', start)            

    def forward(self):
        offsets = torch.tanh(self.offsets)
        normed = offsets / (vector_norm(offsets, dim=1, keepdim=True) + 1e-6)
        vertices = self.start + normed
        return vertices

    
class Layer(nn.Module):
    def __init__(self, start, until, ratio=0.5):
        super(Layer, self).__init__()
        self.ratio = ratio        
        self.offsets = nn.Parameter(torch.zeros_like(start))
        self.register_buffer('start', start)
        self.register_buffer('faces', make_cube_faces(start.size(-1)).int())
        self.register_buffer('scales', self.get_scales(until, self.ratio))
        
    def get_scales(self, vertices, ratio):
        with torch.no_grad():            
            s, n = vertices.size(0), vertices.size(-1)
            pts = vertices.permute(0, 2, 3, 1).reshape(-1, 3)[None]
            dists = knn_points(pts, pts, K=2).dists[:, :, 1]            
            dists = dists.reshape(s, 1, n, n)            
            return dists * ratio

    def forward(self):
        offsets = torch.tanh(self.offsets)
        normed = offsets / (vector_norm(offsets, dim=1, keepdim=True) + 1e-6)
        vertices = self.start + normed * self.scales
        return vertices

class Tree(nn.Module):
    def __init__(self, pts, layers, scale_factor, ratio=0.5):
        super(Tree, self).__init__()
        offsets = treeify(pts, layers, scale_factor)
        untils = get_untils(offsets)
        self.core = Core(offsets[0])
        self.layers = nn.ModuleList([         
            Layer(start, until, ratio=ratio) for (start, until) in zip(offsets[1:], untils[1:])
        ])
        
    def forward(self):
        core = self.core()
        offsets = [layer() for layer in self.layers]
        n = offsets[-1].size(-1)
        res = torch.zeros_like(offsets[-1])
        for t in [core] + offsets:
            res += F.interpolate(t, size=n, mode='nearest')
        return res
