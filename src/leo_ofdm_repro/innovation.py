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


def innovation(current, previous, t, old_t):
    if previous is None:
        return np.zeros((*current.shape[:-1], 8), np.float32)
    dt = t - old_t
    dx = (current[..., 1] - (previous[..., 1] - c.FS / c.FC * (previous[..., 2] * dt + 0.5 * previous[..., 3] * dt * dt))) / (c.FS / 18360000.0)
    df = current[..., 2] - previous[..., 2] - previous[..., 3] * dt
    dr = (current[..., 3] - previous[..., 3]) / 4
    score = np.log1p(np.maximum(current[..., 0], 0)) - np.log1p(np.maximum(previous[..., 0], 0))
    return np.stack((np.clip(dx, -100, 100), np.clip(df, -100, 100), np.clip(dr, -30, 30), score, np.minimum(abs(dx), 100), np.minimum(abs(df), 100), np.minimum(abs(dr), 30), np.ones_like(dx)), -1).astype(np.float32)
