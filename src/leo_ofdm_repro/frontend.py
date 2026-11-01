"""Correlate surveillance and reference samples and estimate noise variances."""
import numpy as np
import torch
import time, math, json, threading
from types import SimpleNamespace
from torch import nn
from torch.nn import functional as F
from scipy import fft
from .parameters import *


def correlate(reference, observation):
    blocks = observation.numel() // BLOCK
    length = fft.next_fast_len(BLOCK + DELAYS - 1)
    windows = torch.as_strided(reference[GUARD - (DELAYS - 1):], (blocks, BLOCK + DELAYS - 1), (BLOCK, 1))
    r = torch.fft.fft(windows, n=length, dim=1).conj()
    o = torch.fft.fft(observation.reshape(blocks, BLOCK), n=length, dim=1)
    lookup = torch.remainder(torch.arange(DELAYS, device=reference.device) - (DELAYS - 1), length)
    return torch.fft.ifft(o * r, dim=1).index_select(1, lookup) / np.sqrt(BLOCK)

def variances5(reference, frames):
    power = reference.real.double().square() + reference.imag.double().square()
    prefix = torch.cat((torch.zeros(1, device=reference.device, dtype=torch.float64), torch.cumsum(power, 0)))
    delays = torch.arange(DELAYS, device=reference.device)
    low = GUARD + torch.arange(frames * 4, device=reference.device)[:, None] * int(0.005 * FS) - delays
    return ((prefix[low + int(0.005 * FS)] - prefix[low]) / int(0.005 * FS)).float()

def complex_maps(correlation, variance5, frame_start=0):
    frames = correlation.shape[0] // PER
    raw = torch.fft.fftshift(torch.fft.fft(correlation.reshape(frames, PER, DELAYS), n=2 * PER, dim=1), dim=1)
    frequencies = fft.fftshift(fft.fftfreq(2 * PER, BLOCK / FS))
    selected = np.flatnonzero(abs(frequencies) <= 1600 + 0.01)
    assert np.array_equal(frequencies[selected], FREQUENCIES)
    variance = variance5.reshape(frames, 4, DELAYS).mean(1)
    z = raw.index_select(1, torch.as_tensor(selected, device=raw.device)) / np.sqrt(PER)
    z /= variance[:, None, :].sqrt()
    centres = (torch.arange(frames, device=raw.device, dtype=torch.float64) + frame_start + 0.5) * 0.02
    focus = torch.exp(-2j * np.pi * centres[:, None] * torch.as_tensor(FREQUENCIES, device=raw.device, dtype=torch.float64)[None])
    z *= focus.to(torch.complex64)[:, :, None]
    return torch.stack((z.real, z.imag, torch.log1p(z.abs().square())), dim=1)
