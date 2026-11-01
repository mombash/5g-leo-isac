"""Selected original numerical operations; provenance and credits in _config/extraction.json and NOTICES.md."""
import numpy as np
import torch
import time, math, json, threading
from types import SimpleNamespace
from torch import nn
from torch.nn import functional as F
from scipy import fft
from .parameters import *
from .model_core import CapacityNet,normalize,flat,cuda_values
from .context import AssetDirectory
OUT=AssetDirectory("models")
old=SimpleNamespace(OLD=AssetDirectory("models"))


def vector(a, rich):
    return np.c_[flat(a, rich), a['observed'].astype(np.float32)]

class Reader:

    def __init__(self, name):
        self.name = name
        self.spec = json.loads((OUT / 'model_specs.json').read_text())[name]
        self.norm = dict(np.load(old.OLD / 'normalization.npz'))
        path = OUT / self.spec['checkpoint']
        if self.spec['family'] == 'neural':
            self.model = CapacityNet(self.spec['rich']).cuda().eval()
            self.model.head[-1] = nn.Linear(128, 28).cuda()
            self.model.load_state_dict(torch.load(path, weights_only=True)['model'])
        elif self.spec['family'] == 'linear':
            self.z = dict(np.load(path))
        else:
            self.model = joblib.load(path)
            self.model.n_jobs = 1

    @torch.no_grad()
    def __call__(self, a):
        if self.spec['family'] == 'neural' and len(a['x']) > 1:
            return np.concatenate([self({k: v[i:i + 1] for (k, v) in a.items()}) for i in range(len(a['x']))])
        z = normalize(a, self.norm)
        if self.spec['family'] == 'neural':
            aa = cuda_values(z)
            result = self.model(*[aa[k] for k in ['x', 'p', 'r', 'm', 'v']]).cpu().numpy()
        elif self.spec['family'] == 'linear':
            x = np.clip((vector(z, self.spec['rich']) - self.z['fm']) / self.z['fs'], -12, 12)
            result = x @ self.z['weight'].T + self.z['bias']
        else:
            result = self.model.predict(vector(z, True))
        result[:, :27][a['observed']] = -1e+30
        return result
