# pyright: reportMissingImports=false
import math
import torch
import torch.nn.functional as F

third = math.pi / 3

def get_angles(a, b, c, dim=1):
    ba = a - b
    bc = c - b    
    dot = (ba * bc).sum(dim=dim)
    norm = torch.linalg.norm(ba, dim=dim) * torch.linalg.norm(bc, dim=dim)
    cosine_angle = dot / norm
    return torch.acos(cosine_angle)# .clip(-1, 1)


def get_angle_loss(t, limit=third):
    a = t[:, :,  :-2, :]
    b = t[:, :, 1:-1, :]
    c = t[:, :, 2:,   :]
    ha = get_angles(a, b, c)
    ha = F.relu(math.pi - ha - limit)
    
    a = t[:, :, :,  :-2]
    b = t[:, :, :, 1:-1]
    c = t[:, :, :, 2:]
    va = get_angles(a, b, c)
    va = F.relu(math.pi - va - limit)    
    res = ha.mean() + va.mean()
    return res * 0.5