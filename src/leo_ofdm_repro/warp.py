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


@triton.jit
def warp_kernel(source, scales, table, destination, columns: tl.constexpr, block: tl.constexpr, taps: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.program_id(1) * block + tl.arange(0, block)
    midpoint = (columns - 1) * 0.5
    scale = tl.load(scales + row)
    coordinate = midpoint + (offsets.to(tl.float64) - midpoint) * scale
    integer = tl.floor(coordinate).to(tl.int32)
    phase = (coordinate - integer.to(tl.float64)) * 4096.0
    index_phase = tl.floor(phase).to(tl.int32)
    fraction = (phase - index_phase.to(tl.float64)).to(tl.float32)
    acc_real = tl.full((block,), 0, tl.float32)
    acc_imag = tl.full((block,), 0, tl.float32)
    for tap in range(taps):
        source_column = integer + tap - (taps // 2 - 1)
        valid = (offsets < columns) & (source_column >= 0) & (source_column < columns)
        address = 2 * (row * columns + source_column)
        real = tl.load(source + address, mask=valid, other=0)
        imag = tl.load(source + address + 1, mask=valid, other=0)
        first = tl.load(table + index_phase * taps + tap, mask=offsets < columns, other=0)
        second = tl.load(table + (index_phase + 1) * taps + tap, mask=offsets < columns, other=0)
        coefficient = first * (1 - fraction) + second * fraction
        acc_real += real * coefficient
        acc_imag += imag * coefficient
    dest_address = 2 * (row * columns + offsets)
    tl.store(destination + dest_address, acc_real, mask=offsets < columns)
    tl.store(destination + dest_address + 1, acc_imag, mask=offsets < columns)

def gpu_warp(source, scales, table):
    (rows, columns) = source.shape
    result = torch.empty_like(source)
    source_float = torch.view_as_real(source).reshape(-1)
    result_float = torch.view_as_real(result).reshape(-1)
    warp_kernel[rows, triton.cdiv(columns, 128)](source_float, scales, table, result_float, columns, 128, 48, num_warps=4)
    return result
