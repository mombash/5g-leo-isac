"""Run dual-scale motion compensation through the stage interface."""
import numpy as np
import torch
import time, math, json, threading
from types import SimpleNamespace
from torch import nn
from torch.nn import functional as F
from scipy import fft
from .parameters import *
from . import dual_scale


class StreamingBackend:

    def __init__(self, batch_rates=2):
        self.batch_rates = batch_rates

    def stage(self, c, v, ordinal, j, k):
        import torch
        if k != 27:
            raise ValueError('Literature dual-scale baseline retains all27 rates')
        n = round(float(TIMES[j]) / 0.02)
        start = ordinal * 50
        with torch.no_grad():
            clock = time.perf_counter()
            out = dual_scale.confirm_rows(c[start * PER:(start + n) * PER], v[start * 4:(start + n) * 4], float(TIMES[j]), self.batch_rates)
            torch.cuda.current_stream().synchronize()
        return dict(j=int(j), k=27, rows=out['rows'], point=out['point'], trace=None, score=None, processing_s=time.perf_counter() - clock)
