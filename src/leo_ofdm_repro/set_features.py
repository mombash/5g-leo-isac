"""Describe searched motion hypotheses for the rate selector."""
import numpy as np
import torch
import time, math, json, threading
from types import SimpleNamespace
from torch import nn
from torch.nn import functional as F
from scipy import fft
from .parameters import *
from .physical_features import physical_features
c=SimpleNamespace(FS=FS,FC=FC)


def observed_features(bank, t):
    bank = np.asarray(bank)
    p = bank[bank[:, 0].argmax()]
    r = bank[:, 3]
    midf = bank[:, 2] - r * t / 2
    midd = bank[:, 1] + c.FS / c.FC * (midf * t / 2 + 0.5 * r * (t / 2) ** 2)
    winner = int(bank[:, 0].argmax())
    dr = r - r.mean()
    den = max(float((dr * dr).sum()), 1.0)
    slope = float((dr * (bank[:, 2] - bank[:, 2].mean())).sum() / den)
    span = max(float(r.max() - r.min()), 4.0)
    edge = float(p[3] == r.min() or p[3] == r.max())
    extra = np.array([edge, abs(p[3]) / 52, (p[3] - r.min()) / span, (r.max() - p[3]) / span, bank[:, 2].std() / 1600, np.clip(bank[:, 1].std() / (c.FS / 18360000.0), 0, 100), span * t * t / 52, edge * t * t, np.clip(slope / (t / 2), -20, 20)], np.float32)
    global_x = np.r_[physical_features(bank, t), extra, len(bank) / 27, t].astype(np.float32)
    row = np.stack((np.log1p(bank[:, 0]), np.log1p(bank[:, 0]) - np.log1p(p[0]), np.clip((midd - midd[winner]) / (c.FS / 18360000.0), -100, 100), np.clip(midf - midf[winner], -100, 100), r / 52, (r - p[3]) / 52, bank[:, 1] / 1792, bank[:, 2] / 1600, np.full(len(bank), t), np.full(len(bank), len(bank) / 27), np.clip(bank[:, 2] - p[2] - (r - p[3]) * t / 2, -100, 100)), axis=-1).astype(np.float32)
    return (row, global_x)
