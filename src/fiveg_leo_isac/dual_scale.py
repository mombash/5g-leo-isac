"""Reuse coarse migration correction across fine Doppler-rate trials."""
import numpy as np
import torch
import time, math, json, threading
from types import SimpleNamespace
from torch import nn
from torch.nn import functional as F
from scipy import fft
from .parameters import *


def approximation_bound(duration, max_rate=52.0, fs=FS, fc=FC):
    u = fs / (2 * fc)
    d = 0.5 * max_rate / fc * (duration / 2) ** 2 / (1 - u)
    return dict(max_delay_samples=d * fs, max_residual_phase_rad=np.pi * max_rate * (duration / 2) ** 2 * u / (1 - u), coarse_half_cell_satisfied=bool(d * fs <= 0.5), midpoint_reference=True, rates_not_pruned=True)

def groups(rates, duration, fs=FS, fc=FC):
    rates = np.asarray(rates, dtype=float)
    radius = (fc - fs / 2) / (fs * max((duration / 2) ** 2, 1e-20))
    coarse = 2 * radius * np.round(rates / (2 * radius))
    return [(float(c), np.flatnonzero(coarse == c)) for c in np.unique(coarse)]

def endpoint(d, f, r, duration, fs=FS, fc=FC):
    t = duration / 2
    return (d - fs / fc * (f * t + 0.5 * r * t * t), f + r * t)

def dual_scale_rows(receiver, start, n, variance, duration, ids=None, batch_rates=2):
    import torch
    from .warp import gpu_warp
    base = receiver.window_fft(start, n)
    (m, ndelay) = base.shape
    if ndelay != DELAYS:
        raise ValueError('production delay grid must remain1792')
    if abs(m * BLOCK * 5 / FS - duration) > 1e-08:
        raise ValueError('duration/sample mismatch')
    if FMAX >= FS / (BLOCK * 5) / 2:
        raise ValueError('ambiguity index q=0 not justified')
    ids = np.arange(len(RATES)) if ids is None else np.asarray(ids, dtype=int)
    if len(np.unique(ids)) != len(ids):
        raise ValueError('duplicate rate indices')
    rr = RATES[ids]
    t = (torch.arange(m, device=base.device, dtype=torch.float64) * BLOCK * 5 + (BLOCK * 5 - 1) / 2) / FS - duration / 2
    warped = gpu_warp(base.T.contiguous(), receiver.scales, receiver.table).T
    nfft = fft.next_fast_len(2 * m)
    frequencies = fft.fftshift(fft.fftfreq(nfft, BLOCK * 5 / FS))
    fi = np.flatnonzero(abs(frequencies) <= FMAX + 0.01)
    findex = torch.as_tensor(fi, device=base.device)
    result = np.empty((len(ids), 5), dtype=float)
    for (coarse, indices) in groups(rr, duration):
        if coarse == 0:
            shared = torch.fft.ifft(warped, dim=1)
        else:
            phase = torch.exp(-1j * np.pi * coarse * t[:, None].square() * receiver.scales[None]).to(base.dtype)
            shared = torch.fft.ifft(warped * phase, dim=1)
        for first in range(0, len(indices), batch_rates):
            ix = indices[first:first + batch_rates]
            fine = torch.as_tensor(rr[ix] - coarse, device=base.device, dtype=torch.float64)
            phase = torch.exp(-1j * np.pi * fine[:, None] * t[None].square()).to(base.dtype)
            z = torch.fft.fftshift(torch.fft.fft(shared[None] * phase[:, :, None], n=nfft, dim=1), dim=1)
            z = z.index_select(1, findex)
            power = z.abs().square() / (m * variance[None, None])
            power[..., :MIN_DELAY] = -float('inf')
            (peaks, where) = power.reshape(len(ix), -1).max(1)
            peaks = peaks.cpu().numpy()
            where = where.cpu().numpy()
            for (j, i) in enumerate(ix):
                f = frequencies[fi[where[j] // ndelay]]
                d = float(where[j] % ndelay)
                (de, fe) = endpoint(d, f, rr[i], duration)
                result[i] = [peaks[j], de, fe, rr[i], len(ids)]
    return result

def numpy_search(warped, t, variance, rates=RATES, mode='dual', fs=FS, fc=FC, minimum_delay=0, fmax=FMAX, duration=None):
    (m, nd) = warped.shape
    dt = float(t[1] - t[0])
    duration = m * dt if duration is None else duration
    nu = fft.fftfreq(nd, 1 / fs)
    scale = fc / (fc + nu)
    nfft = fft.next_fast_len(2 * m)
    freq = fft.fftshift(fft.fftfreq(nfft, dt))
    fi = np.flatnonzero(abs(freq) <= fmax + 0.01)
    out = []
    buckets = groups(rates, duration, fs, fc) if mode == 'dual' else [(float(r), np.array([i])) for (i, r) in enumerate(rates)]
    for (coarse, indices) in buckets:
        shared = fft.ifft(warped * np.exp(-1j * np.pi * coarse * t[:, None] ** 2 * scale[None]), axis=1)
        for i in indices:
            r = float(rates[i])
            signal = shared * np.exp(-1j * np.pi * (r - coarse) * t[:, None] ** 2)
            z = fft.fftshift(fft.fft(signal, n=nfft, axis=0), axes=0)[fi]
            score = abs(z) ** 2 / (m * variance[None])
            score[:, :minimum_delay] = -np.inf
            pos = np.unravel_index(np.argmax(score), score.shape)
            d = float(pos[1])
            f = float(freq[fi[pos[0]]])
            (de, fe) = endpoint(d, f, r, duration, fs, fc)
            out.append((i, [float(score[pos]), de, fe, r, len(rates)]))
    return np.asarray([row for (i, row) in sorted(out)])

def confirm_rows(correlation, variance5, duration, batch_rates=2):
    from .coherent import CoherentEpisode
    from .verifier import LocalVerifier
    from .refinement import classical_refine
    n = round(duration / 0.02)
    receiver = CoherentEpisode(correlation, variance5)
    rows = dual_scale_rows(receiver, 0, n, variance5.mean(0), duration, batch_rates=batch_rates)
    native = rows[rows[:, 0].argmax()].copy()
    point = native.copy()
    verifier = LocalVerifier(correlation, variance5)
    point[:4] = classical_refine(verifier, point[None, 1:4], duration)[0].cpu().numpy()
    return dict(rows=rows, native=native, point=point)
