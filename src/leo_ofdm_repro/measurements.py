"""Assemble observed-rate features and short histories for search selection."""
import numpy as np
import torch
import time, math, json, threading
from types import SimpleNamespace
from torch import nn
from torch.nn import functional as F
from scipy import fft
from .parameters import *
from .set_features import observed_features
from .innovation import innovation
c=SimpleNamespace(FS=FS,FC=FC,TIMES=TIMES)


class Measurements:

    def initial(self, j):
        return np.argsort(self.prior[j])[-3:]

    def state(self, index, j, observed, history=True):
        times = list(range(2, j + 1))[-3:] if history else [j]
        a = dict(x=np.zeros((1, 3, 82), np.float32), p=np.zeros((1, 3, 3, 2, 64), np.float32), r=np.zeros((1, 3, 27, 11), np.float32), m=np.zeros((1, 3, 27), bool), v=np.zeros((1, 3), bool), observed=np.zeros((1, 27), bool))
        previous = previous_t = None
        for (slot, tindex) in enumerate(times):
            ids = np.asarray(observed if tindex == j else self.initial(tindex), int)
            ids = np.sort(ids)
            rows = self.banks[index, tindex, ids]
            (features, global_features) = observed_features(rows, float(c.TIMES[tindex]))
            features[:, 6:8] = 0
            a['r'][0, slot, :len(ids)] = features
            a['m'][0, slot, :len(ids)] = True
            a['x'][0, slot] = self.base[index, tindex]
            a['x'][0, slot, :36] = global_features
            a['x'][0, slot, [19, 20]] = 0
            a['x'][0, slot, 74:] = innovation(self.anchor[index, tindex][None], None if previous is None else previous[None], float(c.TIMES[tindex]), previous_t)[0]
            a['p'][0, slot] = self.trace[index, tindex]
            a['v'][0, slot] = True
            (previous, previous_t) = (self.anchor[index, tindex], float(c.TIMES[tindex]))
        a['observed'][0, observed] = True
        return a
