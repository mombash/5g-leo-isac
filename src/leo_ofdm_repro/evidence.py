"""Selected original numerical operations; provenance and credits in _config/extraction.json and NOTICES.md."""
import numpy as np
import torch
import time, math, json, threading
from types import SimpleNamespace
from torch import nn
from torch.nn import functional as F
from scipy import fft
from .parameters import *
from .set_features import observed_features
from .phase_features import observed_phase_features,midpoint_np
from .innovation import innovation


class Evidence:

    def __init__(self):
        self.reset()

    def reset(self):
        self.x = []
        self.p = []
        self.r = []
        self.m = []
        self.previous = self.previous_t = None

    def append(self, rows, point, trace, score, t, k):
        native = rows[rows[:, 0].argmax()]
        (points, g) = observed_features(rows, t)
        pf = observed_phase_features(trace, native[None], score, t, k)[0]
        base = np.r_[g, pf].astype(np.float32)
        base[[19, 20, 65, 66]] = 0
        old = midpoint_np(native[None, 1:4], t)[0]
        new = midpoint_np(np.asarray(point)[None, 1:4], t)[0]
        delta = np.r_[(new - old) / np.array([2.0, 1 / t, 4.0]), np.log1p(max(point[0], 0)) - np.log1p(max(float(score[0]), 0))]
        ip = innovation(np.asarray(point)[None], None if self.previous is None else self.previous[None], t, self.previous_t)[0]
        self.x.append(np.r_[base, delta, ip].astype(np.float32))
        self.p.append(trace[0])
        points[:, 6:8] = 0
        padded = np.zeros((27, 11), np.float32)
        padded[:k] = points
        self.r.append(padded)
        self.m.append(np.arange(27) < k)
        (self.previous, self.previous_t) = (np.asarray(point).copy(), t)

    def arrays(self):
        a = dict(x=np.zeros((1, 3, 82), np.float32), p=np.zeros((1, 3, 3, 2, 64), np.float32), r=np.zeros((1, 3, 27, 11), np.float32), m=np.zeros((1, 3, 27), bool), v=np.zeros((1, 3), bool))
        for (key, values) in [('x', self.x), ('p', self.p), ('r', self.r), ('m', self.m)]:
            a[key][0, :len(values)] = values
        a['v'][0, :len(self.x)] = True
        return a
