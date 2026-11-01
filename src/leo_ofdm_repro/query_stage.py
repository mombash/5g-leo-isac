"""Selected original numerical operations; provenance and credits in _config/extraction.json and NOTICES.md."""
import numpy as np
import torch
import time, math, json, threading
from types import SimpleNamespace
from torch import nn
from torch.nn import functional as F
from scipy import fft
from .parameters import *
from .measurements import Measurements
from .query_reader import FastReader
from . import final_backend as final
from . import physics as s


class Backend(final.FinalGateBackend):

    def __init__(self, m):
        super().__init__(m)
        self.reader = FastReader(m['query_model']) if m['query_model'] else None
        self.read_lock = threading.Lock()

    @torch.no_grad()
    def stage(self, c, v, ordinal, j, k):
        if j == 6 or self.reader is None or k == 9:
            return super().stage(c, v, ordinal, j, k)
        stream = torch.cuda.current_stream()
        beg = time.perf_counter()
        n = round(float(s.TIMES[j]) / 0.02)
        a = ordinal * 50
        t = float(s.TIMES[j])
        cc = c[a * s.PER:(a + n) * s.PER]
        vv = v[a * 4:(a + n) * 4]
        receiver = s.CoherentEpisode(cc, vv)
        verifier = s.LocalVerifier(cc, vv)
        store = Measurements.__new__(Measurements)
        store.prior = self.prior
        store.banks = np.zeros((1, 7, 27, 5))
        store.base = np.zeros((1, 7, 82), np.float32)
        store.trace = np.zeros((1, 7, 3, 2, 64), np.float32)
        store.anchor = np.zeros((1, 7, 4))
        ids = store.initial(j).copy()
        rows = s.native_rows(receiver, 0, n, vv.mean(0), t, ids=ids, batch_rates=2)
        store.banks[0, j, ids] = rows
        anchor = rows[rows[:, 0].argmax()].copy()
        point = s.classical_refine(verifier, anchor[None, 1:4], t)[0].cpu().numpy()
        (tr, score) = s.phase_traces(verifier, anchor[None, 1:4], n)
        trace = tr.half().float().cpu().numpy()
        evidence = s.runtime.Evidence()
        evidence.append(rows, point, trace, score.cpu().numpy(), t, 3)
        store.base[0, j] = evidence.x[0]
        store.trace[0, j] = trace[0]
        store.anchor[0, j] = point
        state = store.state(0, j, ids, history=False)
        mark = time.perf_counter()
        with self.read_lock:
            ranking = self.reader(state)[0, :27].copy()
            stream.synchronize()
        read_s = time.perf_counter() - mark
        ranking[ids] = -1e+30
        more = np.argsort(ranking)[-2:]
        assert not np.isin(more, ids).any()
        extra = s.native_rows(receiver, 0, n, vv.mean(0), t, ids=more, batch_rates=2)
        rows = np.concatenate([rows, extra])
        best = rows[rows[:, 0].argmax()]
        if not np.array_equal(best[:4], anchor[:4]):
            point = s.classical_refine(verifier, best[None, 1:4], t)[0].cpu().numpy()
        stream.synchronize()
        return dict(j=j, k=k, rows=rows, point=np.r_[point, k], trace=None, score=None, processing_s=time.perf_counter() - beg, selected_ids=np.r_[ids, more].tolist(), readout_s=read_s, final_gate_rejected=False, executed_rates=5)

def looks(m):
    return [(2, m['budgets'][0]), (4, m['budgets'][1]), (6, 27)]
