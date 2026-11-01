"""Decide whether a partial final rate search can reject without an alert."""
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
from .context import AssetDirectory
from .assets import configuration
g=SimpleNamespace(TAU=configuration()["methods"]["neural"]["threshold"])
e=SimpleNamespace(focus=SimpleNamespace(c=SimpleNamespace(OUT=AssetDirectory("policies"))))


def feature(trace, native, score, refined):
    trace = np.asarray(trace, dtype=np.float64)
    u = np.linspace(-1, 1, 64)
    basis = np.stack([np.ones(64), u, 0.5 * (3 * u * u - 1), 0.5 * (5 * u * u * u - 3 * u)])
    moments = np.einsum('cdt,pt->cdp', trace, basis) / 64
    coherent = np.sqrt((trace[1].mean(-1) ** 2).sum())
    moments /= max(coherent, 0.2)
    return np.r_[moments.ravel(), np.log1p((trace ** 2).sum(-2).mean(-1)), np.log1p(max(native[0], 0)), np.log1p(max(float(score), 0)), native[3] / 52, np.log1p(max(refined[0], 0)), np.zeros(4)]

@torch.no_grad()
def run(cc, vv, gate=None, early_warning=False, budget=15):
    torch.cuda.current_stream().synchronize()
    start = time.perf_counter()
    r = CoherentEpisode(cc, vv)
    v = vv.mean(0)
    ver = LocalVerifier(cc, vv)
    if gate is None or early_warning:
        rows = native_rows(r, 0, 50, v, 1.0, ids=np.arange(27), batch_rates=2)
        seed = rows[rows[:, 0].argmax()]
        point = classical_refine(ver, seed[None, 1:4], 1.0)[0].cpu().numpy()
        torch.cuda.current_stream().synchronize()
        return dict(work=time.perf_counter() - start, point=point, skipped=False, rates=27, feature=None)
    ids = np.argsort(np.load(e.focus.c.OUT / 'phase_refinement/rate_prior.npy')[6])[-budget:]
    rows = native_rows(r, 0, 50, v, 1.0, ids=ids, batch_rates=2)
    seed = rows[rows[:, 0].argmax()]
    cheap = classical_refine(ver, seed[None, 1:4], 1.0)[0].cpu().numpy()
    feat = None
    if gate['kind'] == 'physical':
        score = float(cheap[0])
    else:
        (z, q) = phase_traces(ver, seed[None, 1:4], 50)
        z = z.half().float().cpu().numpy()[0]
        q = q.cpu().numpy()[0]
        feat = feature(z, seed, q, cheap)
        score = float(g.prob(gate, dict(x=feat[None], cheap=cheap[None]))[0])
    skip = not early_warning and cheap[0] <= g.TAU and (score < gate['cutoff'])
    cheap_finish = time.perf_counter() - start
    if skip:
        point = cheap
        point[0] = -1e+30
    else:
        remaining = np.setdiff1d(np.arange(27), ids)
        more = native_rows(r, 0, 50, v, 1.0, ids=remaining, batch_rates=2)
        allrows = np.concatenate([rows, more])
        allrows = allrows[np.argsort(allrows[:, 3])]
        best = allrows[allrows[:, 0].argmax()]
        point = cheap if np.array_equal(best[:4], seed[:4]) else classical_refine(ver, best[None, 1:4], 1.0)[0].cpu().numpy()
    torch.cuda.current_stream().synchronize()
    return dict(work=time.perf_counter() - start, cheap_work=cheap_finish, point=point, skipped=bool(skip), rates=budget if skip else 27, feature=feat)
