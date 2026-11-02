"""Schedule fresh-sample sequential detection and final refinement."""
import numpy as np
import torch
import time, math, json, threading
from types import SimpleNamespace
from torch import nn
from torch.nn import functional as F
from scipy import fft
from .parameters import *
from . import scheduler as shared
from .two_step_backend import Backend


class StreamingRecord(shared.StreamingRecord):

    def __init__(self, policy, row, reference, observation, alignment):
        self.policy = policy
        method = dict(id='literature_two_step', kind='two_step', policy={})
        super().__init__(method, Backend(policy), row, reference, observation, alignment, 'pipeline')

    @torch.no_grad()
    def controller(self, ordinal):
        stages = []
        try:
            with torch.cuda.stream(self.controls[ordinal]):
                j = int(np.argmin(abs(shared.TIMES - self.policy.first_s)))
                while True:
                    result = self.request(ordinal, j, 27).result()
                    t = float(shared.TIMES[j])
                    stages.append(dict(prefix_s=t, requested_candidates=27, confirmed_cells=result['confirmed_cells'], processing_s=result['processing_s'], start_s=result['start_s'], finish_s=result['finish_s']))
                    if result['terminal']:
                        finish = time.perf_counter() - self.origin - ordinal
                        self.accepted(ordinal)
                        self.b.states.pop(ordinal, None)
                        point = np.nan_to_num(result['point'], nan=-1e+30, neginf=-1e+30, posinf=-1e+30)
                        out = dict(stop=j, alert=result['alert'], checks=len(stages), prediction=point.tolist(), stages=stages, work=sum((s['processing_s'] for s in stages)), gate_score=result['gate_score'], first_rate_searches=27, fresh_second_cells=sum((s['confirmed_cells'] for s in stages)))
                        return dict(start=ordinal * 50, decision_s=finish, observation_s=t, receiver_processing_s=out['work'], model_control_s=0.0, output=out)
                    j = 6
        except BaseException as exc:
            self.fail(exc)
            raise
