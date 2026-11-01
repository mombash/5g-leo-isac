"""Committed campaign behavior; source mapping in _config/extraction.json."""
import numpy as np
import torch
import time, math, json, threading
from types import SimpleNamespace
from torch import nn
from torch.nn import functional as F
from scipy import fft
from .parameters import *
from . import scheduler as stream
from . import query_stage as query

class QueryRecord(stream.StreamingRecord):

    @torch.no_grad()
    def controller(self, ordinal):
        updates = []
        first = None
        method = self.m
        looks = [(6, 27)] if method.get('noearly') else query.looks(method)
        try:
            with torch.cuda.stream(self.controls[ordinal]):
                futures = [self.request(ordinal, j, k) for (j, k) in looks[:-1]]
                for (pos, (j, k)) in enumerate(looks):
                    if j == 6:
                        self.b.force_full[ordinal] = bool(method.get('always27') or first is not None)
                        future = self.request(ordinal, j, k)
                    else:
                        future = futures[pos]
                    value = future.result()
                    alert = bool(value['point'][0] > method['threshold'])
                    update = dict(j=j, k=k, observation_s=float(stream.TIMES[j]), available_s=time.perf_counter() - self.origin - ordinal, alert=alert, prediction=value['point'].tolist(), processing_s=value['processing_s'], start_s=value['start_s'], finish_s=value['finish_s'], final_gate_rejected=value.get('final_gate_rejected', False), executed_rates=value.get('executed_rates', k), selected_ids=value.get('selected_ids'), readout_s=value.get('readout_s', 0))
                    updates.append(update)
                    if alert and first is None:
                        first = dict(update)
                self.accepted(ordinal)
                return dict(start=ordinal * 50, decision_s=updates[-1]['available_s'], observation_s=1.0, first_warning=first, updates=updates, receiver_processing_s=sum((value['processing_s'] for value in updates)), output=dict(alert=updates[-1]['alert'], prediction=updates[-1]['prediction'], stages=[dict(prefix_s=value['observation_s'], requested_candidates=value['k']) for value in updates]))
        except BaseException as error:
            self.fail(error)
            raise
