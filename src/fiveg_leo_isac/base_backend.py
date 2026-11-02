"""Search specified Doppler rates and refine the strongest peak."""
import numpy as np
import torch
import time, math, json, threading
from types import SimpleNamespace
from torch import nn
from torch.nn import functional as F
from scipy import fft
from .parameters import *
from .coherent import CoherentEpisode
from .verifier import LocalVerifier
from .native import native_rows
from .refinement import classical_refine,phase_traces


class Backend:

    def ids(self, j, k):
        return np.arange(27) if k == 27 else np.argsort(self.prior[j])[-k:]

    @torch.no_grad()
    def stage(self, c, v, ordinal, j, k):
        stream = torch.cuda.current_stream()
        start = time.perf_counter()
        n = round(float(TIMES[j]) / 0.02)
        a = ordinal * 50
        cc = c[a * PER:(a + n) * PER]
        vv = v[a * 4:(a + n) * 4]
        r = CoherentEpisode(cc, vv)
        rows = native_rows(r, 0, n, vv.mean(0), float(TIMES[j]), ids=self.ids(j, k), batch_rates=min(2, k))
        native = rows[rows[:, 0].argmax()].copy()
        point = native.copy()
        point[4] = k
        verifier = LocalVerifier(cc, vv)
        point[:4] = classical_refine(verifier, point[None, 1:4], float(TIMES[j]))[0].cpu().numpy()
        trace = score = None
        if self.model is not None and j < 6:
            (z, s) = phase_traces(verifier, native[None, 1:4], n)
            trace = z.half().float().cpu().numpy()
            score = s.cpu().numpy()
        stream.synchronize()
        return dict(j=j, k=k, rows=rows, point=point, trace=trace, score=score, processing_s=time.perf_counter() - start)
