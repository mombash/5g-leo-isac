"""Selected original numerical operations; provenance and credits in _config/extraction.json and NOTICES.md."""
import numpy as np
import torch
import time, math, json, threading
from types import SimpleNamespace
from torch import nn
from torch.nn import functional as F
from scipy import fft
from .parameters import *


def normalize(a, norm):
    out = dict(a)
    out['x'] = np.where(a['v'][..., None], np.clip((a['x'] - norm['xm']) / norm['xs'], -12, 12), 0).astype(np.float32)
    out['r'] = np.where(a['m'][..., None], np.clip((a['r'] - norm['rm']) / norm['rs'], -12, 12), 0).astype(np.float32)
    out['p'] = np.where(a['v'][..., None, None, None], np.clip((a['p'] - norm['pm'][None, None, :, :, None]) / norm['ps'][None, None, :, :, None], -12, 12), 0).astype(np.float32)
    return out

class CapacityNet(nn.Module):

    def __init__(self, rich=True):
        super().__init__()
        self.rich = rich
        self.summary = nn.Sequential(nn.Linear(82, 96), nn.SiLU(), nn.LayerNorm(96))
        if rich:
            self.row = nn.Sequential(nn.Linear(11, 48), nn.SiLU(), nn.Linear(48, 48), nn.SiLU())
            self.attention = nn.TransformerEncoder(nn.TransformerEncoderLayer(48, 4, 96, dropout=0.1, batch_first=True, norm_first=True), 1, enable_nested_tensor=False)
            self.phase = nn.Sequential(nn.Conv1d(9, 32, 5, padding=2), nn.SiLU(), nn.Conv1d(32, 32, 5, padding=2), nn.SiLU(), nn.Conv1d(32, 32, 3, padding=1), nn.SiLU())
        self.input = nn.Sequential(nn.Linear(96 + (160 if rich else 0), 192), nn.SiLU(), nn.LayerNorm(192))
        self.memory = nn.GRU(192, 192, 2, dropout=0.1, batch_first=True)
        self.head = nn.Sequential(nn.Linear(192, 128), nn.SiLU(), nn.Dropout(0.1), nn.Linear(128, 35))

    def forward(self, x, p, r, m, v):
        (n, length) = x.shape[:2]
        parts = [self.summary(x)]
        if self.rich:
            rr = self.row(r.reshape(n * length, 27, 11))
            mm = m.reshape(n * length, 27).clone()
            empty = ~mm.any(1)
            mm[empty, 0] = True
            rr = self.attention(rr, src_key_padding_mask=~mm)
            count = mm.sum(1).clamp_min(1)[:, None]
            pooled = torch.cat(((rr * mm[..., None]).sum(1) / count, rr.masked_fill(~mm[..., None], -10000.0).amax(1)), -1)
            pooled[empty] = 0
            pp = torch.cat((p[:, :, :, 0], p[:, :, :, 1], torch.log1p(p.square().sum(3))), 2).reshape(n * length, 9, 64)
            zz = self.phase(pp)
            zz = torch.cat((zz.mean(-1), zz.amax(-1)), -1)
            parts += [pooled.reshape(n, length, 96), zz.reshape(n, length, 64)]
        z = self.input(torch.cat(parts, -1))
        z = self.memory(z)[0]
        last = v.long().sum(1) - 1
        return self.head(z[torch.arange(n, device=x.device), last])

def flat(a, rich):
    pieces = [a['x'].reshape(len(a['x']), -1), a['v'].astype(np.float32)]
    if rich:
        ph = np.concatenate((a['p'][:, :, :, 0], a['p'][:, :, :, 1], np.log1p((a['p'] ** 2).sum(3))), 2)
        pieces += [a['r'].reshape(len(a['x']), -1), a['m'].reshape(len(a['x']), -1).astype(np.float32), ph.reshape(len(a['x']), -1)]
    return np.concatenate(pieces, -1).astype(np.float32)

def cuda_values(a):
    return {k: torch.from_numpy(a[k]).cuda() for k in ['x', 'p', 'r', 'm', 'v', 'y', 'allowed'] if k in a}
