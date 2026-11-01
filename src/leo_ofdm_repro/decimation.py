"""Selected original numerical operations; provenance and credits in _config/extraction.json and NOTICES.md."""
import numpy as np
import torch
import time, math, json, threading
from types import SimpleNamespace
from torch import nn
from torch.nn import functional as F
from scipy import fft
from .parameters import *
from scipy.signal import firwin
FILTER=firwin(193,1950,fs=20000,window=('kaiser',8.6)).astype(np.float32)


def decimate_gpu(compressed, kernel):
    (count, delays) = compressed.shape
    real_imag = torch.view_as_real(compressed).permute(1, 2, 0).contiguous()
    shifted = torch.empty_like(real_imag)
    shifted[:, :, :-2] = real_imag[:, :, 2:]
    shifted[:, :, -2:] = 0
    batched = shifted.reshape(delays * 2, 1, count)
    filtered = F.conv1d(F.pad(batched, (96, 96)), kernel, stride=5)
    result = filtered.reshape(delays, 2, count // 5).permute(2, 0, 1).contiguous() * np.sqrt(5.0)
    return torch.view_as_complex(result)
