# pyright: reportMissingImports=false
import torch

def encode(t, freq, steps, log=True):        
    bands = 2.**torch.linspace(0., freq, steps) if log \
        else torch.linspace(1., 2.**freq, steps)
    bands = bands[None, None, :, None, None]
    sin = torch.sin(t[:, :, None] * bands)
    cos = torch.cos(t[:, :, None] * bands)
    res = torch.cat([sin, cos], dim=2)
    return res.reshape(6, -1, t.size(-1) , t.size(-1))
