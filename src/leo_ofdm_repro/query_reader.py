"""Selected original numerical operations; provenance and credits in _config/extraction.json and NOTICES.md."""
import numpy as np
import torch
import time, math, json, threading
from types import SimpleNamespace
from torch import nn
from torch.nn import functional as F
from scipy import fft
from .parameters import *
from .reader_core import Reader,vector,normalize
from .model_static import StaticNet,Captured,Affine


class FastReader(Reader):

    def __init__(self, name):
        super().__init__(name)
        if self.spec['family'] == 'neural':
            net = StaticNet(self.spec['rich']).cuda().eval()
            net.head[-1] = nn.Linear(128, 28).cuda()
            net.load_state_dict(self.model.state_dict())
            for parameter in net.parameters():
                parameter.requires_grad_(False)
            dummy = [torch.zeros((1, 3, 82), device='cuda'), torch.zeros((1, 3, 3, 2, 64), device='cuda'), torch.zeros((1, 3, 27, 11), device='cuda'), torch.zeros((1, 3, 27), device='cuda', dtype=torch.bool), torch.zeros((1, 3), device='cuda', dtype=torch.bool)]
            dummy[-1][:, 0] = True
            dummy[-2][:, 0, :3] = True
            self.captured = Captured(net, dummy)
        elif self.spec['family'] == 'linear':
            model = Affine(torch.from_numpy(self.z['weight']).cuda(), torch.from_numpy(self.z['bias']).cuda())
            self.captured = Captured(model, [torch.zeros((1, len(self.z['fm'])), device='cuda')])

    @torch.no_grad()
    def __call__(self, a):
        z = normalize(a, self.norm)
        if self.spec['family'] == 'neural':
            values = [torch.from_numpy(z[k]).cuda() for k in ['x', 'p', 'r', 'm', 'v']]
            result = self.captured(*values).cpu().numpy().copy()
        elif self.spec['family'] == 'linear':
            x = np.clip((vector(z, self.spec['rich']) - self.z['fm']) / self.z['fs'], -12, 12)
            result = self.captured(torch.from_numpy(x).cuda()).cpu().numpy().copy()
        else:
            result = self.model.predict(vector(z, True))
        result[:, :27][a['observed']] = -1e+30
        return result
