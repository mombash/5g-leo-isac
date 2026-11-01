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


def midpoint_np(p, t):
    out = p.copy()
    dt = t / 2
    out[..., 1] = p[..., 1] - dt * p[..., 2]
    out[..., 0] = p[..., 0] + c.FS / c.FC * (dt * out[..., 1] + 0.5 * dt * dt * p[..., 2])
    return out

def endpoint_np(p, t):
    out = p.copy()
    dt = t / 2
    out[..., 1] = p[..., 1] + dt * p[..., 2]
    out[..., 0] = p[..., 0] - c.FS / c.FC * (dt * p[..., 1] + 0.5 * dt * dt * p[..., 2])
    return out

def observed_phase_features(trace, native, score, ts, ks):
    trace = np.asarray(trace, dtype=np.float32)
    native = np.asarray(native)
    shape = trace.shape[:-3]
    ts = np.broadcast_to(ts, shape)
    ks = np.broadcast_to(ks, shape)
    u = np.linspace(-1, 1, 64, dtype=np.float32)
    basis = np.stack((np.ones(64), u, 0.5 * (3 * u * u - 1), 0.5 * (5 * u * u * u - 3 * u)), 0).astype(np.float32)
    moments = np.einsum('...cdt,pt->...cdp', trace, basis) / 64
    coherent = np.sqrt((trace[..., 1, :, :].mean(-1) ** 2).sum(-1))
    moments /= np.maximum(coherent, 0.2)[..., None, None, None]
    moments[..., 2] /= ts[..., None, None] ** 2
    moments = moments.reshape(*shape, 24)
    energy = np.log1p((trace ** 2).sum(-2).mean(-1))
    scalar = np.stack((np.log1p(np.maximum(native[..., 0], 0)), np.log1p(np.maximum(score, 0)), native[..., 1] / 1792, native[..., 2] / 1600, native[..., 3] / 52, ts, ks / 27), -1)
    return np.concatenate((moments, energy, scalar), -1).astype(np.float32)
