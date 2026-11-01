"""Selected original numerical operations; provenance and credits in _config/extraction.json and NOTICES.md."""
import numpy as np
import torch
import time, math, json, threading
from types import SimpleNamespace
from torch import nn
from torch.nn import functional as F
from scipy import fft
from .parameters import *
c=SimpleNamespace(FS=FS,FC=FC)


def physical_features(bank, t):
    p = bank[bank[:, 0].argmax()]
    a = np.log1p(bank[:, 0])
    order = np.sort(a)[::-1]
    order = np.pad(order, (0, max(0, 10 - len(order))), mode='edge')[:10]
    close = (abs(bank[:, 1] - p[1]) <= c.FS / 18360000.0) & (abs(bank[:, 2] - p[2]) <= 1)
    return np.r_[order, order[0] - order[1:], p[1] / 1792, p[2] / 1600, p[3] / 52, t, close.mean(), a.std()].astype(np.float32)
