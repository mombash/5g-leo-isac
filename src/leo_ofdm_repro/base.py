"""Initialize a fixed-budget physical-search backend."""
import threading
import numpy as np
from .base_backend import Backend
from .context import current


class BaseBackend(Backend):
    def __init__(self,method):
        if method['kind']!='simple':raise ValueError('Expected a fixed-budget search policy.')
        self.m=method
        self.prior=None if method['policy']['k']==27 else np.load(current().policies/'phase_refinement/rate_prior.npy')
        self.model=None
        self.model_lock=threading.Lock()
        self.cost=None
