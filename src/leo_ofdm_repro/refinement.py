"""Selected original numerical operations; provenance and credits in _config/extraction.json and NOTICES.md."""
import numpy as np
import torch
import time, math, json, threading
from types import SimpleNamespace
from torch import nn
from torch.nn import functional as F
from scipy import fft
from .parameters import *


def endpoint(mid, t):
    p = mid.clone()
    dt = t / 2
    p[:, 1] = mid[:, 1] + dt * mid[:, 2]
    p[:, 0] = mid[:, 0] - FS / FC * (dt * mid[:, 1] + 0.5 * dt * dt * mid[:, 2])
    return p

def midpoint(end, t):
    p = end.clone()
    dt = t / 2
    p[:, 1] = end[:, 1] - dt * end[:, 2]
    p[:, 0] = end[:, 0] + FS / FC * (dt * p[:, 1] + 0.5 * dt * dt * end[:, 2])
    return p

@torch.no_grad()
def phase_traces(verifier, parameters, frames, bins=64):
    p = torch.as_tensor(parameters, device=verifier.c.device, dtype=torch.float64)
    offset = torch.tensor([-1.0, 0.0, 1.0], device=p.device)
    points = p[:, None].expand(-1, 3, -1).clone()
    points[:, :, 0] += offset
    (signal, variance, valid) = verifier.sample(points.reshape(-1, 3), 0, frames, False)
    n = signal.shape[-1]
    edges = torch.linspace(0, n, bins + 1, device=p.device).long()
    integral = torch.cat((torch.zeros((len(signal), 1), device=p.device, dtype=signal.dtype), signal.cumsum(-1)), -1)
    summed = integral[:, edges[1:]] - integral[:, edges[:-1]]
    normalizer = (variance[:, None] * (edges[1:] - edges[:-1])[None] / n).clamp_min(1e-08).sqrt()
    z = summed / normalizer
    z = z.reshape(len(p), 3, bins)
    anchor = z[:, 1].sum(-1)
    aligned = z * anchor.conj()[:, None, None] / anchor.abs().clamp_min(1e-08)[:, None, None]
    trace = torch.stack((aligned.real, aligned.imag), 2).float()
    score = signal.sum(-1).abs().square() / variance.clamp_min(1e-08)
    score = score.reshape(len(p), 3)[:, 1]
    score = score.masked_fill(~valid.reshape(len(p), 3)[:, 1], -1e+30)
    return (trace, score)

def physical_scores(verifier, parameters, frames):
    value = verifier.sample(parameters, 0, frames, True)
    allowed = (parameters[:, 0] >= 80) & (parameters[:, 0] <= 1791) & (parameters[:, 1].abs() <= 1600) & (parameters[:, 2].abs() <= 52)
    return value.masked_fill(~allowed, -float('inf'))

@torch.no_grad()
def classical_refine(verifier, seeds, t, iterations=6):
    p = midpoint(torch.as_tensor(seeds, device=verifier.c.device, dtype=torch.float64), t)
    origin = p.clone()
    q = len(p)
    frames = round(t / 0.02)
    radius = p.new_tensor([2.0, 1.0 / t, 4.0])
    steps = p.new_tensor([0.25, 0.25 / t, 1.0])
    offsets = torch.cat((p.new_zeros((1, 3)), torch.diag(steps), -torch.diag(steps)))
    best = physical_scores(verifier, endpoint(p, t), frames)
    for _ in range(iterations):
        test = torch.minimum(torch.maximum(p[:, None] + offsets[None], origin[:, None] - radius), origin[:, None] + radius)
        test[:, :, 2].clamp_(-52, 52)
        values = physical_scores(verifier, endpoint(test.reshape(-1, 3), t), frames).reshape(q, 7)
        left = values[:, 4:]
        centre = values[:, :1]
        right = values[:, 1:4]
        den = left - 2 * centre + right
        correction = torch.where((den < -1e-07) & torch.isfinite(left) & torch.isfinite(centre) & torch.isfinite(right), 0.5 * (left - right) / den, torch.zeros_like(den)).clamp(-1, 1) * steps
        proposed = torch.minimum(torch.maximum(p + correction, origin - radius), origin + radius)
        proposed[:, 2].clamp_(-52, 52)
        pv = physical_scores(verifier, endpoint(proposed, t), frames)
        i = values.argmax(1)
        grid = test[torch.arange(q, device=p.device), i]
        gv = values[torch.arange(q, device=p.device), i]
        better = pv > gv
        candidate = torch.where(better[:, None], proposed, grid)
        value = torch.maximum(pv, gv)
        accepted = value > best
        p = torch.where(accepted[:, None], candidate, p)
        best = torch.maximum(value, best)
    return torch.cat((torch.nan_to_num(best, nan=-1e+30, neginf=-1e+30, posinf=-1e+30)[:, None], endpoint(p, t)), 1)
