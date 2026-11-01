"""Selected original numerical operations; provenance and credits in _config/extraction.json and NOTICES.md."""
import numpy as np
import torch
import time, math, json, threading
from types import SimpleNamespace
from torch import nn
from torch.nn import functional as F
from scipy import fft
from .parameters import *
from .model_core import CapacityNet


class StaticNet(CapacityNet):

    def forward(self, x, p, r, m, v):
        (n, length) = x.shape[:2]
        parts = [self.summary(x)]
        if self.rich:
            rr = self.row(r.reshape(n * length, 27, 11))
            mm = m.reshape(n * length, 27)
            empty = ~mm.any(1)
            mm = mm.clone()
            mm[:, 0] = mm[:, 0] | empty
            rr = self.attention(rr, src_key_padding_mask=~mm)
            count = mm.sum(1).clamp_min(1)[:, None]
            pooled = torch.cat(((rr * mm[..., None]).sum(1) / count, rr.masked_fill(~mm[..., None], -10000.0).amax(1)), -1)
            pooled = torch.where(empty[:, None], torch.zeros_like(pooled), pooled)
            pp = torch.cat((p[:, :, :, 0], p[:, :, :, 1], torch.log1p(p.square().sum(3))), 2).reshape(n * length, 9, 64)
            zz = self.phase(pp)
            zz = torch.cat((zz.mean(-1), zz.amax(-1)), -1)
            parts += [pooled.reshape(n, length, 96), zz.reshape(n, length, 64)]
        z = self.memory(self.input(torch.cat(parts, -1)))[0]
        last = v.long().sum(1) - 1
        return self.head(z[torch.arange(n, device=x.device), last])

class Affine(nn.Module):

    def __init__(self, w, b):
        super().__init__()
        self.register_buffer('w', w)
        self.register_buffer('b', b)

    def forward(self, x):
        return x @ self.w.T + self.b

class Captured:

    def __init__(self, model, inputs):
        self.model = model
        self.inputs = [x.clone() for x in inputs]
        self.graph = None
        self.error = None
        with torch.no_grad():
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    model(*self.inputs)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            try:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    self.output = model(*self.inputs)
                self.graph = graph
            except RuntimeError as e:
                self.error = str(e)

    @torch.no_grad()
    def __call__(self, *inputs):
        if self.graph is None:
            return self.model(*inputs)
        for (dst, src) in zip(self.inputs, inputs):
            dst.copy_(src)
        self.graph.replay()
        return self.output
