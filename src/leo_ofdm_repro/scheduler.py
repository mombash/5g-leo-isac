"""Release samples and schedule preparation, searches and receiver outputs."""
import numpy as np
import torch
import time, math, json, threading
from types import SimpleNamespace
from torch import nn
from torch.nn import functional as F
from scipy import fft
from .parameters import *
import concurrent.futures as cf
from .frontend import correlate,variances5
from .cancellation import cancel_sinc
from . import evidence as runtime
_STREAM_RESOURCES=None


def stream_resources():
    global _STREAM_RESOURCES
    if _STREAM_RESOURCES is None:
        _STREAM_RESOURCES = dict(search=[torch.cuda.Stream() for _ in range(2)], prep=torch.cuda.Stream(), controls=[torch.cuda.Stream() for _ in range(4)])
    return _STREAM_RESOURCES

class StreamingRecord:

    def __init__(self, m, b, row, reference, observation, alignment, mode):
        self.m = m
        self.b = b
        self.row = row
        self.reference = reference
        self.observation = observation
        self.alignment = alignment
        self.mode = mode
        self.c = torch.zeros((200 * PER, 1792), dtype=torch.complex64, device='cuda')
        self.v = torch.zeros((800, 1792), device='cuda')
        torch.cuda.synchronize()
        self.condition = threading.Condition()
        self.ready = 0
        self.jobs = {}
        self.done = set()
        self.prep = []
        self.executed = []
        self.failed = None
        self.stop = False
        self.pool = cf.ThreadPoolExecutor(max_workers=2, thread_name_prefix='physical-search')
        self.streams = stream_resources()['search']
        self.slots = list(range(2))
        self.prep_stream = stream_resources()['prep']
        self.controls = stream_resources()['controls']
        self.backend_timers = []

    def request(self, ordinal, j, k, speculative=False):
        key = (ordinal, j, k)
        with self.condition:
            if key not in self.jobs:
                self.jobs[key] = dict(future=cf.Future(), submitted=False, speculative=speculative)
            self.condition.notify_all()
            return self.jobs[key]['future']

    def fail(self, exc):
        with self.condition:
            self.failed = exc
            self.stop = True
            for x in self.jobs.values():
                if not x['future'].done():
                    x['future'].set_exception(exc)
            self.condition.notify_all()

    @torch.no_grad()
    def prepare(self):
        try:
            with torch.cuda.stream(self.prep_stream):
                cursor = 0
                while cursor < 200:
                    with self.condition:
                        while True:
                            if self.stop or len(self.done) == 4:
                                return
                            ordinal = cursor // 50
                            if ordinal in self.done:
                                cursor = (ordinal + 1) * 50
                                if cursor >= 200:
                                    return
                                continue
                            full = 10 * (cursor // 10 + 1)
                            requested = [key[0] * 50 + round(float(TIMES[key[1]]) / 0.02) for (key, x) in self.jobs.items() if key[0] not in self.done and (not x['future'].done())]
                            possible = [v for v in requested if cursor < v < full]
                            end = min(possible) if possible else full
                            delay = self.origin + end * 0.02 - time.perf_counter()
                            if delay <= 0:
                                break
                            self.condition.wait(delay)
                    beg = time.perf_counter()
                    base = 10 * ((end - 1) // 10)
                    available = end - base
                    rr = self.reference[base * COUNT:GUARD + end * COUNT].copy()
                    oo = self.observation[base * COUNT:end * COUNT].copy()
                    al = np.zeros((10, 2))
                    al[:available] = self.alignment[base:end]
                    if available < 10:
                        rr = np.r_[rr, np.tile(rr[-COUNT:], 10 - available)]
                        oo = np.r_[oo, np.zeros((10 - available) * COUNT, np.complex64)]
                    rx = torch.from_numpy(rr).cuda()
                    obs = torch.from_numpy(oo).cuda()
                    (clean, _) = cancel_sinc(rx, obs, al)
                    vv = variances5(rx, 10)
                    cc = correlate(rx, clean)
                    first = cursor
                    offset = first - base
                    self.c[first * PER:end * PER] = cc[offset * PER:available * PER]
                    self.v[first * 4:end * 4] = vv[offset * 4:available * 4]
                    self.prep_stream.synchronize()
                    finished = time.perf_counter()
                    self.prep.append(dict(first_frame=first, last_frame=end, source_batch=base, available_frames=available, ready_s=end * 0.02, start_s=beg - self.origin, finish_s=finished - self.origin, processing_s=finished - beg))
                    with self.condition:
                        self.ready = end
                        cursor = end
                        self.condition.notify_all()
                    del rr, oo, rx, obs, cc, vv, clean
        except BaseException as exc:
            self.fail(exc)

    @torch.no_grad()
    def execute(self, key, slot):
        job = self.jobs[key]
        beg = time.perf_counter()
        try:
            with torch.cuda.stream(self.streams[slot]):
                value = self.b.stage(self.c, self.v, *key)
            finish = time.perf_counter()
            value['start_s'] = beg - self.origin
            value['finish_s'] = finish - self.origin
            with self.condition:
                self.executed.append(dict(ordinal=key[0], j=key[1], k=key[2], start_s=value['start_s'], finish_s=value['finish_s'], processing_s=value['processing_s'], speculative=job['speculative'], slot=slot))
                self.slots.append(slot)
                if not job['future'].done():
                    job['future'].set_result(value)
                self.condition.notify_all()
        except BaseException as exc:
            self.fail(exc)

    def dispatch(self):
        try:
            while True:
                with self.condition:
                    if self.failed:
                        return
                    pending = [(key, x) for (key, x) in self.jobs.items() if not x['submitted'] and (not x['future'].done()) and (key[0] not in self.done) and (key[0] * 50 + round(float(TIMES[key[1]]) / 0.02) <= self.ready)]
                    pending.sort(key=lambda z: (z[0][0] + float(TIMES[z[0][1]]), z[1]['speculative']))
                    for (key, job) in pending:
                        if not self.slots:
                            break
                        slot = self.slots.pop()
                        job['submitted'] = True
                        self.pool.submit(self.execute, key, slot)
                    if len(self.done) == 4 or self.stop:
                        return
                    self.condition.wait(0.002)
        except BaseException as exc:
            self.fail(exc)

    def accepted(self, ordinal):
        with self.condition:
            self.done.add(ordinal)
            for (key, job) in self.jobs.items():
                if key[0] == ordinal and (not job['submitted']) and (not job['future'].done()):
                    job['future'].cancel()
            self.condition.notify_all()

    @torch.no_grad()
    def controller(self, ordinal):
        m = self.m
        p = m['policy']
        state = runtime.Evidence()
        stages = []
        modelwork = 0.0
        lc = rc = 0.0
        last = -1
        try:
            with torch.cuda.stream(self.controls[ordinal]):
                if m['kind'] == 'simple':
                    looks = list(p['looks'])
                    j = looks[0]
                    k = 27 if j == 6 else p['k']
                else:
                    j = p.get('first_j', 2)
                    k = p['first_k']
                if self.mode == 'parallel':
                    if m['kind'] == 'simple':
                        for jj in looks[1:]:
                            self.request(ordinal, jj, 27 if jj == 6 else p['k'], True)
                    elif j < 6:
                        self.request(ordinal, 6, 27, True)
                while True:
                    result = self.request(ordinal, j, k).result()
                    point = result['point']
                    t = float(TIMES[j])
                    stages.append(dict(prefix_s=t, requested_candidates=k, processing_s=result['processing_s'], start_s=result['start_s'], finish_s=result['finish_s']))
                    if m['kind'] == 'simple':
                        eligible = True
                        following = None if j == looks[-1] else (looks[looks.index(j) + 1], p['k'])
                    elif j == 6:
                        eligible = True
                        following = None
                    else:
                        beg = time.perf_counter()
                        state.append(result['rows'], point, result['trace'], result['score'], t, k)
                        q = self.b.predict(state)
                        if m.get('schedule_variant'):
                            lc = max(lc, t) + self.b.cost.legacy[j, e.KS.index(k)]
                            rc = float(self.b.cost.step(rc, last, j, k))
                            (ok, nxt) = neural.choose(q, p, j, k, len(stages), np.array([lc]), np.array([rc]), self.b.cost)
                        else:
                            lc = max(lc, t) + action_costs(0.003)[j * 5 + e.KS.index(k)]
                            (ok, nxt) = runtime.decide(q, p, j, k, len(stages), lc)
                        eligible = bool(ok[0])
                        action = int(nxt[0])
                        following = (action // 5, e.KS[action % 5])
                        modelwork += time.perf_counter() - beg
                    alert = bool(eligible and point[0] > m['threshold'])
                    if alert or following is None or j == 6:
                        finish = time.perf_counter() - self.origin - ordinal
                        self.accepted(ordinal)
                        out = dict(stop=j, alert=alert, checks=len(stages), candidates=sum((s['requested_candidates'] for s in stages)), prediction=point.tolist(), stages=stages, work=sum((s['processing_s'] for s in stages)) + modelwork)
                        return dict(start=ordinal * 50, decision_s=finish, observation_s=t, receiver_processing_s=out['work'], model_control_s=modelwork, output=out)
                    last = j
                    (j, k) = following
                    k = 27 if j == 6 else k
        except BaseException as exc:
            self.fail(exc)
            raise

    def run(self):
        self.origin = time.perf_counter()
        prep = threading.Thread(target=self.prepare)
        dispatch = threading.Thread(target=self.dispatch)
        prep.start()
        dispatch.start()
        with cf.ThreadPoolExecutor(max_workers=4, thread_name_prefix='controller') as pool:
            futures = [pool.submit(self.controller, i) for i in range(4)]
            attempts = [f.result() for f in futures]
        with self.condition:
            self.stop = True
            self.condition.notify_all()
        prep.join()
        dispatch.join()
        self.pool.shutdown(wait=True)
        torch.cuda.synchronize()
        if self.failed:
            raise self.failed
        for a in attempts:
            a['shared_processing_s'] = sum((s['processing_s'] for s in self.prep if a['start'] <= s['first_frame'] < a['start'] + 50))
        used = {(i, j['prefix_s'], j['requested_candidates']) for (i, a) in enumerate(attempts) for j in a['output']['stages']}
        for x in self.executed:
            x['used_by_policy'] = (x['ordinal'], float(TIMES[x['j']]), x['k']) in used
        return dict(attempts=attempts, shared_stages=self.prep, executed_searches=self.executed, drained_wall_s=time.perf_counter() - self.origin, max_search_workers=2)
