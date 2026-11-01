"""Explicit initialization for the selected simple physical-search branch."""
import threading
import numpy as np
from .base_backend import Backend
from .context import current


class BaseBackend(Backend):
    def __init__(self,method):
        if method['kind']!='simple':raise ValueError('Only selected simple search policies are supported.')
        self.m=method
        self.prior=None if method['policy']['k']==27 else np.load(current().policies/'phase_refinement/rate_prior.npy')
        self.model=None
        self.model_lock=threading.Lock()
        # The original constructor also instantiated historical timing-cost
        # forecasts. They are never consulted by this selected stage branch.
        self.cost=None
