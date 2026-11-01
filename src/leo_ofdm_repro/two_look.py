"""Selected original numerical operations; provenance and credits in _config/extraction.json and NOTICES.md."""
import numpy as np
import torch
import time, math, json, threading
from types import SimpleNamespace
from torch import nn
from torch.nn import functional as F
from scipy import fft
from .parameters import *
from dataclasses import dataclass


@dataclass(frozen=True)
class Policy:
    first_s: float
    second_s: float
    lower: float
    upper: float
    final: float

    def __post_init__(self):
        if not (0 < self.first_s <= 1 and 0 <= self.second_s and (self.first_s + self.second_s <= 1.000000001)):
            raise ValueError('Two fresh looks must fit within the one-second attempt.')
        if not (0 <= self.lower <= self.upper and 0 <= self.final):
            raise ValueError('Require 0 <= lower <= upper and final >= 0.')
        if self.second_s == 0 and self.lower != self.upper:
            raise ValueError('Zero second dwell requires the fixed-one-look case lower=upper.')
        for t in (self.first_s, self.second_s):
            if not np.isclose(t / 0.02, round(t / 0.02)):
                raise ValueError('Dwell durations must align to 20 ms frames.')

def first_decisions(y1, policy):
    y1 = np.asarray(y1, float)
    if not np.isfinite(y1).all() or (y1 < 0).any():
        raise ValueError('Eq.(10) energies must be finite and nonnegative.')
    return dict(negative=y1 < policy.lower, positive=y1 >= policy.upper, revisit=(y1 >= policy.lower) & (y1 < policy.upper))

def cell_decisions(y1, y2, policy):
    (y1, y2) = (np.asarray(y1, float), np.asarray(y2, float))
    if y1.shape != y2.shape:
        raise ValueError('Cell identities must match across looks.')
    d = first_decisions(y1, policy)
    if not np.isfinite(y2[d['revisit']]).all() or (y2[d['revisit']] < 0).any():
        raise ValueError('Requested second-look energies must be finite and nonnegative.')
    d['confirmed'] = d['revisit'] & (y2 >= policy.final)
    d['detected'] = d['positive'] | d['confirmed']
    return d

def propagate(parameters, delta_s, fs=FS, fc=FC):
    p = np.asarray(parameters, float).copy()
    p[..., 0] -= fs / fc * (p[..., 1] * delta_s + 0.5 * p[..., 2] * delta_s ** 2)
    p[..., 1] += p[..., 2] * delta_s
    return p

def normalized_energy(samples, variance):
    z = np.asarray(samples)
    v = np.asarray(variance, float)
    if np.any(v <= 0):
        raise ValueError('Noise variance must be positive.')
    return abs(z.sum(-1)) ** 2 / np.broadcast_to(v, z.shape).sum(-1)

def run(policy, first_search, second_search, *, wait_until=None, refine=None):
    if wait_until:
        wait_until(policy.first_s)
    first = first_search(policy.first_s, policy.lower, policy.upper)
    stages = [dict(start_s=0.0, duration_s=policy.first_s, work_s=first['work_s'])]
    positive = first.get('best_positive_row')
    alerts = np.asarray(first['alert_parameters'], float).reshape(-1, 3)
    if positive is not None:
        row = np.asarray(positive, float)
        if row[0] < policy.upper:
            raise ValueError('First positive below upper threshold.')
        answer = dict(alert=True, look=1, decision_s=policy.first_s, row=row, tested_second_cells=0, gate_score=float(row[0]))
    elif not len(alerts):
        answer = dict(alert=False, look=1, decision_s=policy.first_s, row=None, tested_second_cells=0, gate_score=None)
    else:
        end = policy.first_s + policy.second_s
        if wait_until:
            wait_until(end)
        second = second_search(policy.first_s, policy.second_s, propagate(alerts, policy.second_s))
        stages.append(dict(start_s=policy.first_s, duration_s=policy.second_s, work_s=second['work_s']))
        row = np.asarray(second['best_row'], float)
        answer = dict(alert=bool(row[0] >= policy.final), look=2, decision_s=end, row=row, tested_second_cells=len(alerts), gate_score=float(row[0]))
    if answer['alert'] and refine is not None:
        start = 0.0
        duration = answer['decision_s']
        (row, seconds) = refine(start, duration, answer['row'])
        answer['row'] = np.asarray(row, float)
        stages.append(dict(start_s=start, duration_s=duration, refinement=True, work_s=seconds))
    answer['stages'] = stages
    answer['work_s'] = sum((s['work_s'] for s in stages))
    return answer

def gpu_callbacks(h, variance, *, cell_batch=2048):
    import time
    import torch
    from scipy import fft
    from .coherent import CoherentEpisode
    from .frontend import PER
    from .parameters import BLOCK, DELAYS, MIN_DELAY, FMAX
    from .warp import gpu_warp
    from .verifier import LocalVerifier
    from .refinement import classical_refine
    rates = np.arange(-52.0, 52.01, 4.0)

    @torch.no_grad()
    def first_search(duration, lower, upper):
        torch.cuda.current_stream().synchronize()
        begin = time.perf_counter()
        frames = round(duration / 0.02)
        cc = h[:frames * PER].cuda()
        vv = variance[:frames * 4].cuda()
        receiver = CoherentEpisode(cc, vv)
        base = receiver.window_fft(0, frames)
        m = len(base)
        clock = (torch.arange(m, device=cc.device, dtype=torch.float64) * BLOCK * 5 + (BLOCK * 5 - 1) / 2) / FS - duration / 2
        nfft = fft.next_fast_len(2 * m)
        freq = fft.fftshift(fft.fftfreq(nfft, BLOCK * 5 / FS))
        fi = np.flatnonzero(abs(freq) <= FMAX + 0.01)
        selected = torch.as_tensor(fi, device=cc.device)
        axis = torch.as_tensor(freq[fi], device=cc.device)
        positive = None
        best_row = None
        alert_chunks = []
        alert_scores = []
        for first in range(0, len(rates), 4):
            rr = rates[first:first + 4]
            r = torch.as_tensor(rr, device=cc.device, dtype=torch.float64)
            phase = torch.exp(-1j * np.pi * r[:, None] * clock.square()).to(torch.complex64)
            source = (base.T[None] * phase[:, None]).reshape(-1, m).contiguous()
            warped = gpu_warp(source, receiver.scales.repeat(len(rr)), receiver.table)
            corrected = torch.fft.ifft(warped.reshape(len(rr), DELAYS, m).transpose(1, 2), dim=2)
            z = torch.fft.fftshift(torch.fft.fft(corrected, n=nfft, dim=1), dim=1).index_select(1, selected)
            score = z.abs().square() / (m * vv.mean(0)[None, None])
            score[..., :MIN_DELAY] = -float('inf')

            def coordinates(index):
                (q, f, d) = index.unbind(-1)
                mid = torch.stack((d.double(), axis[f], r[q]), -1)
                dt = duration / 2
                end = mid.clone()
                end[:, 0] -= FS / FC * (mid[:, 1] * dt + 0.5 * mid[:, 2] * dt ** 2)
                end[:, 1] += mid[:, 2] * dt
                return end
            (peak, idx) = score.reshape(-1).max(0)
            n = int(idx)
            ind = torch.tensor([[n // (len(fi) * DELAYS), n // DELAYS % len(fi), n % DELAYS]], device=cc.device)
            row = np.r_[float(peak), coordinates(ind)[0].cpu().numpy()]
            if best_row is None or row[0] > best_row[0]:
                best_row = row
            if float(peak) >= upper:
                if positive is None or row[0] > positive[0]:
                    positive = row
            if positive is None:
                index = torch.nonzero((score >= lower) & (score < upper))
                if len(index):
                    alert_chunks.append(coordinates(index).cpu().numpy())
                    alert_scores.append(score[index[:, 0], index[:, 1], index[:, 2]].cpu().numpy())
        alerts = np.concatenate(alert_chunks) if positive is None and alert_chunks else np.empty((0, 3))
        torch.cuda.current_stream().synchronize()
        return dict(best_positive_row=positive, best_row=best_row, alert_parameters=alerts, alert_scores=np.concatenate(alert_scores) if positive is None and alert_scores else np.empty(0), work_s=time.perf_counter() - begin)

    @torch.no_grad()
    def second_search(start, duration, parameters, return_cell_scores=False):
        torch.cuda.current_stream().synchronize()
        begin = time.perf_counter()
        first = round(start / 0.02)
        frames = round(duration / 0.02)
        cc = h[first * PER:(first + frames) * PER].cuda()
        vv = variance[first * 4:(first + frames) * 4].cuda()
        verifier = LocalVerifier(cc, vv)
        best = None
        cell_scores = []
        for offset in range(0, len(parameters), cell_batch):
            p = torch.as_tensor(parameters[offset:offset + cell_batch], device=cc.device, dtype=torch.float64)
            values = verifier.sample(p, 0, frames, True)
            i = int(values.argmax())
            row = np.r_[float(values[i]), p[i].cpu().numpy()]
            if return_cell_scores:
                cell_scores.append(values.cpu().numpy())
            if best is None or row[0] > best[0]:
                best = row
        if best is None:
            raise ValueError('No requested second-look cells.')
        torch.cuda.current_stream().synchronize()
        result = dict(best_row=best, work_s=time.perf_counter() - begin)
        if return_cell_scores:
            result['cell_scores'] = np.concatenate(cell_scores)
        return result

    @torch.no_grad()
    def refine(start, duration, row):
        torch.cuda.current_stream().synchronize()
        begin = time.perf_counter()
        first = round(start / 0.02)
        frames = round(duration / 0.02)
        cc = h[first * PER:(first + frames) * PER].cuda()
        vv = variance[first * 4:(first + frames) * 4].cuda()
        result = classical_refine(LocalVerifier(cc, vv), row[None, 1:4], duration)[0].cpu().numpy()
        torch.cuda.current_stream().synchronize()
        return (result, time.perf_counter() - begin)
    return (first_search, second_search, refine)
