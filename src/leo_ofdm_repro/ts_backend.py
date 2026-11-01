"""Execute the fixed-look or fresh-second-look sequential detector."""
import numpy as np
import torch
import time, math, json, threading
from types import SimpleNamespace
from torch import nn
from torch.nn import functional as F
from scipy import fft
from .parameters import *
from . import two_look as detector


class Backend:

    def __init__(self, policy):
        self.policy = policy
        self.states = {}

    @torch.no_grad()
    def stage(self, c, v, ordinal, j, k):
        (first, second, refine) = detector.gpu_callbacks(c[ordinal * 20000:(ordinal + 1) * 20000], v[ordinal * 200:(ordinal + 1) * 200])
        p = self.policy
        if j != 6 or p.second_s == 0:
            one = first(p.first_s, p.lower, p.upper)
            self.states[ordinal] = one
            row = one.get('best_positive_row')
            positive = row is not None
            if row is None:
                row = one['best_row']
            work = one['work_s']
            terminal = positive or not len(one['alert_parameters'])
            if positive:
                (row, refine_work) = refine(0.0, p.first_s, np.asarray(row))
                work += refine_work
            return dict(point=np.asarray(row), processing_s=work, terminal=terminal, alert=positive, confirmed_cells=0, gate_score=float(one['best_row'][0]))
        one = self.states.pop(ordinal)
        value = second(p.first_s, p.second_s, detector.propagate(one['alert_parameters'], p.second_s))
        row = np.asarray(value['best_row'])
        positive = bool(row[0] >= p.final)
        work = value['work_s']
        gate = float(row[0])
        if positive:
            (row, refine_work) = refine(0.0, p.first_s + p.second_s, row)
            work += refine_work
        return dict(point=row, processing_s=work, terminal=True, alert=positive, confirmed_cells=len(one['alert_parameters']), gate_score=gate)
