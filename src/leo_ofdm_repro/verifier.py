"""Evaluate and refine local coherent delay–Doppler hypotheses."""
import numpy as np
import torch
import time, math, json, threading
from types import SimpleNamespace
from torch import nn
from torch.nn import functional as F
from scipy import fft
from .parameters import *
import triton
import triton.language as tl
from .warp_table import interpolation_table


@triton.jit
def _sample(C, V, P, T, OUT, PART, COUNT, N: tl.constexpr, FIRST: tl.constexpr, VFIRST: tl.constexpr, TOTAL: tl.constexpr, PIECES: tl.constexpr, REDUCE: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    part = tl.program_id(1)
    n = part * B + tl.arange(0, B)
    d = tl.load(P + row * 3)
    fd = tl.load(P + row * 3 + 1)
    rate = tl.load(P + row * 3 + 2)
    dt = (n.to(tl.float64) + 0.5) * (1536.0 / 30720000.0) - N * (1536.0 / 30720000.0) + 1.0 / 30720000.0
    coordinate = d - 30720000.0 / 2190000000.0 * (fd * dt + 0.5 * rate * dt * dt)
    integer = tl.floor(coordinate).to(tl.int32)
    phase = (coordinate - integer) * 4096.0
    pi = tl.minimum(tl.floor(phase).to(tl.int32), 4095)
    fraction = (phase - pi).to(tl.float32)
    good = (n < N) & (coordinate >= 80) & (coordinate <= 1791)
    real = tl.full((B,), 0.0, tl.float32)
    imag = tl.full((B,), 0.0, tl.float32)
    for tap in range(48):
        index = integer + tap - 23
        valid = good & (index >= 0) & (index < 1792)
        address = 2 * (index * TOTAL + FIRST + n)
        cr = tl.load(C + address, valid, other=0.0)
        ci = tl.load(C + address + 1, valid, other=0.0)
        a = tl.load(T + pi * 48 + tap, n < N, other=0.0)
        b = tl.load(T + (pi + 1) * 48 + tap, n < N, other=0.0)
        coefficient = a + (b - a) * fraction
        real += cr * coefficient
        imag += ci * coefficient
    angle = (-2.0 * math.pi * (fd * dt + 0.5 * rate * dt * dt)).to(tl.float32)
    co = tl.cos(angle)
    si = tl.sin(angle)
    rr = real * co - imag * si
    ii = real * si + imag * co
    vv = tl.load(V + (FIRST + n) // 100 * 1792 + integer, good, other=0.0)
    norm = tl.sum(vv, 0)
    count = tl.sum(good.to(tl.int32), 0)
    tl.store(COUNT + row * PIECES + part, count)
    if REDUCE:
        tl.store(PART + 3 * (row * PIECES + part), tl.sum(rr, 0))
        tl.store(PART + 3 * (row * PIECES + part) + 1, tl.sum(ii, 0))
        tl.store(PART + 3 * (row * PIECES + part) + 2, norm)
    else:
        tl.store(OUT + 2 * (row * N + n), rr, n < N)
        tl.store(OUT + 2 * (row * N + n) + 1, ii, n < N)
        tl.store(PART + row * PIECES + part, norm)

class LocalVerifier:

    def __init__(self, correlation, variance):
        self.c = correlation.transpose(0, 1).contiguous()
        self.v = variance
        self.table = torch.from_numpy(interpolation_table()).to(correlation.device)

    def sample(self, parameters, start, frames, reduce):
        n = frames * 400
        q = len(parameters)
        pieces = triton.cdiv(n, 256)
        parameters = parameters.to(device=self.c.device, dtype=torch.float64).contiguous()
        output = torch.empty((q, n), device=self.c.device, dtype=torch.complex64) if not reduce else torch.empty(1, device=self.c.device)
        partial = torch.empty((q, pieces, 3) if reduce else (q, pieces), device=self.c.device)
        counts = torch.empty((q, pieces), device=self.c.device, dtype=torch.int32)
        _sample[q, pieces](torch.view_as_real(self.c).reshape(-1), self.v.reshape(-1), parameters, self.table, torch.view_as_real(output).reshape(-1) if not reduce else output, partial, counts, n, start * 400, start * 4, self.c.shape[1], pieces, reduce, 256, num_warps=4)
        valid = counts.sum(1) == n
        if reduce:
            sums = partial.sum(1)
            score = (sums[:, 0].square() + sums[:, 1].square()) / sums[:, 2].clamp_min(1e-08)
            return score.masked_fill(~valid, -float('inf'))
        return (output, partial.sum(1), valid)

    @torch.no_grad()
    def verify(self, proposals, start, frames, full_rate=False):
        torch.cuda.synchronize()
        began = time.perf_counter()
        p = torch.as_tensor(proposals, device=self.c.device, dtype=torch.float64)
        k = len(p)
        dd = torch.arange(-3.0, 3.1, 1.0, device=p.device)
        ff = torch.tensor([-25.0, 0.0, 25.0], device=p.device)
        rr = torch.cat((torch.tensor([-52.0], device=p.device), torch.arange(-48.0, 48.1, 8.0, device=p.device), torch.tensor([52.0], device=p.device))) if full_rate else torch.arange(-8.0, 8.1, 4.0, device=p.device)
        grid = torch.cartesian_prod(dd, ff, rr)
        parameters = p[:, None, 1:4] + grid[None]
        parameters[:, :, 0].clamp_(MIN_DELAY, DELAYS - 1)
        parameters[:, :, 1].clamp_(-1600, 1600)
        parameters[:, :, 2].clamp_(-52, 52)
        flat = parameters.reshape(-1, 3)
        (signal, var, valid) = self.sample(flat, start, frames, False)
        n = frames * 400
        nfft = 2 * n
        freq = torch.fft.fftfreq(nfft, BLOCK / FS, device=p.device)
        indices = torch.nonzero(abs(freq) <= 12.5 + 0.001).flatten()
        axis = freq[indices]
        spectrum = torch.fft.fft(signal, n=nfft, dim=1).index_select(1, indices)
        score = spectrum.abs().square() / var[:, None].clamp_min(1e-08)
        score.masked_fill_(~valid[:, None], -float('inf'))
        score.masked_fill_((flat[:, 1, None] + axis[None]).abs() > 1600, -float('inf'))
        best = score.reshape(k, -1).argmax(1)
        per = len(grid) * len(axis)
        row = best // len(axis) + torch.arange(k, device=p.device) * len(grid)
        fi = best % len(axis)
        centres = flat[row].clone()
        centres[:, 1] += axis[fi]
        del signal, spectrum, score
        fine_extent = 4.0 if full_rate else 2.0
        fine_grid = torch.cartesian_prod(torch.arange(-1.0, 1.01, 0.25, device=p.device), torch.arange(-2.0, 2.01, 1 / (2 * frames * 0.02), device=p.device), torch.arange(-fine_extent, fine_extent + 0.01, 1.0, device=p.device))
        fine = centres[:, None] + fine_grid[None]
        fine[:, :, 0].clamp_(MIN_DELAY, DELAYS - 1)
        fine[:, :, 1].clamp_(-1600, 1600)
        fine[:, :, 2].clamp_(-52, 52)
        refined = self.sample(fine.reshape(-1, 3), start, frames, True).reshape(k, -1)
        j = refined.argmax(1)
        selected = fine[torch.arange(k, device=p.device), j]
        values = refined[torch.arange(k, device=p.device), j]
        self.last_valid_proposals = torch.isfinite(values).cpu().numpy()
        values = torch.nan_to_num(values, nan=-1e+30, neginf=-1e+30, posinf=-1e+30)
        result = torch.cat((values[:, None], selected), 1)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - began
        return (result.float().cpu().numpy(), elapsed)

def cpu_scores(c, v, parameters, start, frames):
    table = interpolation_table()
    n = frames * 400
    indices = np.arange(n)
    dt = (indices + 0.5) * BLOCK / FS - n * BLOCK / FS + 1 / FS
    result = []
    for (d, fd, rate) in parameters:
        coord = d - FS / FC * (fd * dt + 0.5 * rate * dt * dt)
        integer = np.floor(coord).astype(int)
        ph = (coord - integer) * 4096
        pi = np.minimum(np.floor(ph).astype(int), 4095)
        fraction = ph - pi
        value = np.zeros(n, dtype=np.complex128)
        for tap in range(48):
            at = integer + tap - 23
            valid = (at >= 0) & (at < DELAYS)
            weights = table[pi, tap] + (table[pi + 1, tap] - table[pi, tap]) * fraction
            value += c[start * 400 + indices, np.clip(at, 0, DELAYS - 1)] * weights * valid
        value *= np.exp(-2j * np.pi * (fd * dt + 0.5 * rate * dt * dt))
        variance = v[(start * 400 + indices) // 100, integer].sum()
        result.append(abs(value.sum()) ** 2 / variance)
    return np.asarray(result)
