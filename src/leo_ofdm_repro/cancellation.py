"""Selected original numerical operations; provenance and credits in _config/extraction.json and NOTICES.md."""
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
def _template(R, A, T, O, N: tl.constexpr, COUNT_: tl.constexpr, GUARD_: tl.constexpr, OFFSET: tl.constexpr, B: tl.constexpr):
    n = tl.program_id(0) * B + tl.arange(0, B)
    frame = n // COUNT_
    local = n % COUNT_
    a = tl.load(A + 2 * frame, n < N, other=0.0)
    b = tl.load(A + 2 * frame + 1, n < N, other=0.0)
    ds = a + b * (local.to(tl.float64) / 30720000.0 - 0.01)
    coordinate = GUARD_ + n.to(tl.float64) + ds + OFFSET
    integer = tl.floor(coordinate).to(tl.int64)
    ph = (coordinate - integer) * 4096.0
    pi = tl.minimum(tl.floor(ph).to(tl.int32), 4095)
    fraction = (ph - pi).to(tl.float32)
    real = tl.full((B,), 0.0, tl.float32)
    imag = tl.full((B,), 0.0, tl.float32)
    latest = GUARD_ + (frame + 1) * COUNT_ - 1
    for tap in range(48):
        at = integer + tap - 23
        valid = (n < N) & (at >= 0) & (at <= latest)
        rr = tl.load(R + 2 * at, valid, other=0.0)
        ii = tl.load(R + 2 * at + 1, valid, other=0.0)
        ca = tl.load(T + pi * 48 + tap, n < N, other=0.0)
        cb = tl.load(T + (pi + 1) * 48 + tap, n < N, other=0.0)
        weight = ca + (cb - ca) * fraction
        real += weight * rr
        imag += weight * ii
    phase = (2 * math.pi * (2190000000.0 / 30720000.0) * ds).to(tl.float32)
    co = tl.cos(phase)
    si = tl.sin(phase)
    tl.store(O + 2 * n, real * co - imag * si, n < N)
    tl.store(O + 2 * n + 1, real * si + imag * co, n < N)

def cancel_sinc(reference, observation, alignment, offsets=(0,)):
    frames = observation.numel() // COUNT
    n = frames * COUNT
    a = torch.as_tensor(alignment, device=reference.device, dtype=torch.float64).contiguous()
    table = torch.from_numpy(interpolation_table()).to(reference.device)
    templates = []
    for offset in offsets:
        out = torch.empty(n, device=reference.device, dtype=torch.complex64)
        _template[triton.cdiv(n, 256),](torch.view_as_real(reference).reshape(-1), a, table, torch.view_as_real(out).reshape(-1), n, COUNT, GUARD, offset, 256)
        templates.append(out.reshape(frames, COUNT))
    x = torch.stack(templates, 1)
    local = torch.linspace(-1, 1, COUNT, device=reference.device)
    design = torch.cat((x, x * local[None, None]), 1)
    design[:, :, -32:] = 0
    received = observation.reshape(frames, COUNT)
    gram = (design.conj() @ design.transpose(1, 2)).to(torch.complex128) / COUNT
    rhs = (design.conj() @ received[:, :, None]).to(torch.complex128) / COUNT
    regularization = gram.diagonal(dim1=1, dim2=2).real.mean(1) * 1e-07
    gram += regularization[:, None, None] * torch.eye(gram.shape[-1], device=reference.device)[None]
    coefficients = torch.linalg.solve(gram, rhs).to(torch.complex64)
    fitted = (design.transpose(1, 2) @ coefficients)[:, :, 0]
    clean = received - fitted
    clean[:, -32:] = 0
    return (clean.reshape(-1), {'offsets': list(offsets), 'discarded_samples_per_20ms': 32})
