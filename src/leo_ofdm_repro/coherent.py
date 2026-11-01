"""Selected original numerical operations; provenance and credits in _config/extraction.json and NOTICES.md."""
import numpy as np
import torch
import time, math, json, threading
from types import SimpleNamespace
from torch import nn
from torch.nn import functional as F
from scipy import fft
from .parameters import *
from .decimation import decimate_gpu,FILTER
from .warp_table import interpolation_table
from .warp import gpu_warp
def rates():return RATES.tolist()


def decode(score, index, frequencies, duration, rate=0, refine=None):
    i = int(index)
    fi = i // DELAYS
    di = i % DELAYS
    f = float(frequencies[fi])
    d = float(di)
    if refine is not None:
        if 0 < fi < len(frequencies) - 1:
            (a, b, c) = [float(refine[fi + j, di]) for j in (-1, 0, 1)]
            den = a - 2 * b + c
            if np.isfinite([a, b, c]).all() and den < 0:
                f += float(np.clip(0.5 * (a - c) / den, -0.5, 0.5)) * (frequencies[1] - frequencies[0])
    dt = duration / 2
    return {'score': float(score), 'midpoint_delay_samples': d, 'midpoint_frequency_Hz': f, 'delay_samples': float(d - FS / FC * (f * dt + 0.5 * rate * dt ** 2)), 'frequency_Hz': float(f + rate * dt), 'rate_Hz_s': float(rate)}

class ClassicalEpisode:

    def window_fft(self, start, frames):
        a = start * PER
        n = frames * PER
        m = n // 5
        result = self.range_fft[a // 5:a // 5 + m].clone()
        lo = decimate_gpu(self.c[a:a + 500], self.kernel)[:24]
        hi = decimate_gpu(self.c[a + n - 500:a + n], self.kernel)[-24:]
        result[:24] = torch.fft.fft(lo, dim=1)
        result[-24:] = torch.fft.fft(hi, dim=1)
        return result

class CoherentEpisode(ClassicalEpisode):

    def __init__(self, c, v5):
        self.c = c
        self.v5 = v5
        self.kernel = torch.from_numpy(FILTER).to(c.device).reshape(1, 1, -1)
        self.table = torch.from_numpy(interpolation_table()).to(c.device)
        self.scales = torch.from_numpy(np.ascontiguousarray(FC / (FC + np.fft.fftfreq(DELAYS, 1 / FS)), dtype=np.float64)).to(c.device)
        self.rates = rates()
        self.reduced = decimate_gpu(c, self.kernel)
        self.range_fft = torch.fft.fft(self.reduced, dim=1)
