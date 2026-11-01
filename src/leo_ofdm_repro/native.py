"""Evaluate the native motion-compensated Doppler-rate grid."""
import numpy as np
import torch
import time, math, json, threading
from types import SimpleNamespace
from torch import nn
from torch.nn import functional as F
from scipy import fft
from .parameters import *
from .coherent import decode
from .warp import gpu_warp
pc=SimpleNamespace(RATES=RATES)
def batches():return {}


@torch.no_grad()
def native_rows(receiver, start, n, variance, duration, ids=None, batch_rates=None):
    ids = np.arange(len(pc.RATES)) if ids is None else np.asarray(ids, dtype=int)
    batch_rates = int(batch_rates if batch_rates is not None else batches().get(str(float(duration)), {}).get(str(len(ids)), 4))
    base = receiver.window_fft(start, n)
    m = base.shape[0]
    clock = (torch.arange(m, device=base.device, dtype=torch.float64) * BLOCK * 5 + (BLOCK * 5 - 1) / 2) / FS - duration / 2
    nfft = fft.next_fast_len(2 * m)
    f = fft.fftshift(fft.fftfreq(nfft, BLOCK * 5 / FS))
    fi = np.flatnonzero(abs(f) <= FMAX + 0.01)
    selected = torch.as_tensor(fi, device=base.device)
    result = []
    for first in range(0, len(ids), batch_rates):
        rr = pc.RATES[ids[first:first + batch_rates]]
        rate = torch.as_tensor(rr, device=base.device, dtype=torch.float64)
        phase = torch.exp(-1j * np.pi * rate[:, None] * clock.square()[None]).to(torch.complex64)
        source = (base.T[None] * phase[:, None]).reshape(-1, m).contiguous()
        warped = gpu_warp(source, receiver.scales.repeat(len(rr)), receiver.table)
        corrected = torch.fft.ifft(warped.reshape(len(rr), DELAYS, m).transpose(1, 2), dim=2)
        z = torch.fft.fftshift(torch.fft.fft(corrected, n=nfft, dim=1), dim=1).index_select(1, selected)
        values = z.abs().square() / (m * variance[None, None])
        values[..., :MIN_DELAY] = -float('inf')
        (peaks, indices) = values.reshape(len(rr), -1).max(1)
        for (i, r) in enumerate(rr):
            p = decode(peaks[i], indices[i], f[fi], duration, float(r))
            result.append([p['score'], p['delay_samples'], p['frequency_Hz'], p['rate_Hz_s'], len(ids)])
    return np.asarray(result)
