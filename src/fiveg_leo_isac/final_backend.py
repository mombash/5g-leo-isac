"""Apply the final rejection gate and full-search continuation."""
import numpy as np
import torch
import time, math, json, threading
from types import SimpleNamespace
from torch import nn
from torch.nn import functional as F
from scipy import fft
from .parameters import *
from .base import BaseBackend
from .gate import run
from .context import AssetDirectory
stream=SimpleNamespace(Backend=BaseBackend,PER=PER)
gate_data=SimpleNamespace(OUT=AssetDirectory("policies/final_gate"))


class FinalGateBackend(stream.Backend):

    def __init__(self, m, gate_kind='physical'):
        super().__init__(m)
        self.gate = next((g for g in json.loads((gate_data.OUT / 'freeze.json').read_text())['selected'] if g['kind'] == gate_kind))
        self.force_full = {}

    @torch.no_grad()
    def stage(self, c, v, ordinal, j, k):
        if j != 6 or self.force_full.get(ordinal, True):
            out = super().stage(c, v, ordinal, j, k)
            out.update(final_gate_rejected=False, executed_rates=k)
            return out
        a = ordinal * 50
        cc = c[a * stream.PER:(a + 50) * stream.PER]
        vv = v[a * 4:(a + 50) * 4]
        result = run(cc, vv, self.gate, False, budget=15)
        point = np.r_[result['point'], result['rates']]
        return dict(j=j, k=k, rows=np.empty((0, 5)), point=point, trace=None, score=None, processing_s=result['work'], final_gate_rejected=result['skipped'], executed_rates=result['rates'])
