"""Estimate motion using keystone alignment and Lv’s distribution."""
import numpy as np
import torch
import time, math, json, threading
from types import SimpleNamespace
from torch import nn
from torch.nn import functional as F
from scipy import fft
from .parameters import *


def interpolate_fft(x, factor=8):
    import torch
    n = x.shape[0]
    nn = fft.next_fast_len(n + 256)
    y = torch.zeros((nn, x.shape[1]), dtype=x.dtype, device=x.device)
    y[128:128 + n] = x
    a = torch.fft.fftshift(torch.fft.fft(y, dim=0), dim=0)
    z = torch.zeros((nn * factor, x.shape[1]), dtype=x.dtype, device=x.device)
    lo = (nn * factor - nn) // 2
    z[lo:lo + nn] = a
    if nn % 2 == 0 and factor > 1:
        z[lo] = a[0] / 2
        z[lo + nn] = a[0] / 2
    return torch.fft.ifft(torch.fft.ifftshift(z, dim=0), dim=0) * factor

def coordinates(n, dt, nt=64, oversample=8, device='cuda'):
    import torch
    T = n * dt
    lag = torch.arange(1, n + 1, device=device, dtype=torch.float64) * dt
    du = T * T / (4 * nt)
    u = (torch.arange(nt, device=device, dtype=torch.float64) - nt / 2) * du
    t = u[:, None] / lag[None]
    q1 = ((t + lag[None] / 2 + T / 2) / dt - 0.5 + 128) * oversample
    q2 = ((t - lag[None] / 2 + T / 2) / dt - 0.5 + 128) * oversample
    valid = t.abs() + lag[None] / 2 <= T / 2 - dt / 2
    q1 = q1.clamp(0, (n + 255) * oversample - 2)
    q2 = q2.clamp(0, (n + 255) * oversample - 2)
    i1 = q1.floor().long()
    i2 = q2.floor().long()
    return (i1, i2, (q1 - i1).float(), (q2 - i2).float(), valid, du)

def support_plan(n, dt, nt, oversample, device, lag_block):
    import torch
    (i1, i2, w1, w2, mask, du) = coordinates(n, dt, nt, oversample, device)
    norm = max(float(mask.sum()), 1.0)
    T = n * dt
    u = (np.arange(nt) - nt / 2) * du
    chunks = []
    for first in range(0, n, lag_block):
        last = min(n, first + lag_block)
        lag = np.arange(first + 1, last + 1) * dt
        bound = max(float(np.max(lag * (T / 2 - dt / 2 - lag / 2))), 0.0)
        lo = max(0, int(np.searchsorted(u, -bound)) - 1)
        hi = min(nt, int(np.searchsorted(u, bound, side='right')) + 1)
        sl = (slice(lo, hi), slice(first, last))
        chunks.append((first, last, lo, hi, *[a[sl].contiguous() for a in (i1, i2, w1, w2, mask)]))
    return (chunks, norm, du)

def lvd(x, dt, variance=None, min_delay=0, batch=8, nt=64, oversample=8, return_cube=False, time_transform='matrix', lag_block=256):
    import torch
    (n, nd) = x.shape
    T = n * dt
    (chunks, norm, du) = support_plan(n, dt, nt, oversample, x.device, lag_block)
    nr = 2 * nt
    nf = fft.next_fast_len(2 * n)
    rr = fft.fftshift(fft.fftfreq(nr, du))
    ff = fft.fftshift(fft.fftfreq(nf, dt))
    ri = np.flatnonzero(abs(rr) <= 52 + 1e-09)
    fi = np.flatnonzero(abs(ff) <= 1600 + 1e-09)
    rix = torch.as_tensor(ri, device=x.device)
    fix = torch.as_tensor(fi, device=x.device)
    matrix = torch.exp(-2j * torch.pi * torch.as_tensor(rr[ri], device=x.device, dtype=torch.float64)[:, None] * torch.arange(nt, device=x.device, dtype=torch.float64)[None] * du).to(x.dtype) if time_transform == 'matrix' else None
    peaks = []
    positions = []
    patches = []
    widths = []
    starts = []
    cubes = []
    for first in range(min_delay, nd, batch):
        end = min(nd, first + batch)
        z = interpolate_fft(x[:, first:end], oversample)
        from .lag_product import product
        assert time_transform == 'matrix', 'Support optimization is validated for the retained matrix transform only'
        a = torch.empty((len(ri), n, end - first), device=x.device, dtype=x.dtype)
        for (lf, ll, lo, hi, i1, i2, w1, w2, mask) in chunks:
            small = product(z, i1, i2, w1, w2, mask)
            a[:, lf:ll] = (matrix[:, lo:hi] @ small.reshape(hi - lo, -1)).reshape(len(ri), ll - lf, end - first)
        a = torch.fft.fftshift(torch.fft.fft(a, n=nf, dim=1), dim=1).index_select(1, fix)
        power = a.abs().square() / norm ** 2
        if variance is not None:
            power /= variance[None, None, first:end].square()
        (peak, pos) = power.reshape(-1).max(0)
        ir = pos // (len(fi) * (end - first))
        jf = pos // (end - first) % len(fi)
        kd = pos % (end - first)
        offsets = torch.arange(-1, 2, device=x.device)
        patch = power[(ir + offsets).clamp(0, len(ri) - 1)[:, None, None], (jf + offsets).clamp(0, len(fi) - 1)[None, :, None], (kd + offsets).clamp(0, end - first - 1)[None, None, :]]
        peaks.append(peak)
        positions.append(pos)
        patches.append(patch)
        widths.append(end - first)
        starts.append(first)
        if return_cube:
            cubes.append(power.cpu().numpy())
    scores = torch.stack(peaks)
    winner = scores.argmax()
    packed = torch.cat([scores[winner, None].double(), torch.stack(positions)[winner, None].double(), winner[None].double(), torch.stack(patches)[winner].reshape(-1).double()]).cpu().numpy()
    (score, pos, wi) = packed[:3]
    pos = int(pos)
    wi = int(wi)
    width = widths[wi]
    ir = pos // (len(fi) * width)
    jf = pos // width % len(fi)
    kd = pos % width
    patch = packed[3:].reshape(3, 3, 3)
    patch = patch[1 if ir == 0 else 0:2 if ir == len(ri) - 1 else 3, 1 if jf == 0 else 0:2 if jf == len(fi) - 1 else 3, 1 if kd == 0 else 0:2 if kd == width - 1 else 3].astype(np.float32)
    best = dict(score=float(score), delay=float(starts[wi] + kd), frequency=float(ff[fi[jf]]), rate=float(rr[ri[ir]]))
    patch = np.log(np.maximum(patch, 1e-30))
    centre = list(np.unravel_index(np.argmax(patch), patch.shape))
    for (axis, name, step, inner) in [(0, 'rate', rr[1] - rr[0], 0 < ir < len(ri) - 1), (1, 'frequency', ff[1] - ff[0], 0 < jf < len(fi) - 1)]:
        if not inner:
            continue
        ix = centre.copy()
        ix[axis] -= 1
        aa = patch[tuple(ix)]
        ix[axis] += 1
        bb = patch[tuple(ix)]
        ix[axis] += 1
        cc = patch[tuple(ix)]
        den = aa - 2 * bb + cc
        if den < 0:
            best[name] += float(np.clip(0.5 * (aa - cc) / den, -0.5, 0.5) * step)
    return (best, np.concatenate(cubes, axis=2), rr[ri], ff[fi]) if return_cube else best

def infer(c, v, T=1.0, nt=None, batch=8, time_transform='matrix', lag_block=256):
    import torch
    from .coherent import CoherentEpisode
    from .warp import gpu_warp
    st = time.perf_counter()
    receiver = CoherentEpisode(c, v)
    base = receiver.window_fft(0, round(T / 0.02))
    warped = gpu_warp(base.T.contiguous(), receiver.scales, receiver.table).T
    x = torch.fft.ifft(warped, dim=1)
    q = lvd(x, T / len(x), v.mean(0), min_delay=80, nt=len(x) if nt is None else nt, batch=batch, time_transform=time_transform, lag_block=lag_block)
    d = q['delay'] - FS / FC * (q['frequency'] * T / 2 + 0.5 * q['rate'] * (T / 2) ** 2)
    f = q['frequency'] + q['rate'] * T / 2
    torch.cuda.current_stream().synchronize()
    return dict(point=[q['score'], d, f, q['rate']], midpoint=q, processing_s=time.perf_counter() - st)
