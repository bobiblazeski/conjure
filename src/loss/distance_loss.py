# pyright: reportMissingImports=false
import torch
import torch.nn.functional as F

def get_ratios(a, b, c, threshold):
    ba = a - b
    bc = c - b
    bal = ba.abs().sum(dim=1)    
    bcl = bc.abs().sum(dim=1)
    mx = torch.where(bal >= bcl, bal, bcl)
    mn = torch.where(bal < bcl, bal, bcl)
    ratio = mx / mn
    return F.relu(ratio - threshold)
    
def get_distance_loss(t, threshold=1.5):
    a = t[:, :,  :-2, :]
    b = t[:, :, 1:-1, :]
    c = t[:, :, 2:,   :]    
    hr = get_ratios(a, b, c, threshold)
    
    a = t[:, :, :,  :-2]
    b = t[:, :, :, 1:-1]
    c = t[:, :, :, 2:]
    vr = get_ratios(a, b, c, threshold)    
    res = hr.mean() + vr.mean()
    return res * 0.5