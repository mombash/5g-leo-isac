"""Generate deterministic complex Gaussian noise from recorded seeds."""
import numpy as np
from .scenes import seed

def noise(count, identity):
    r = np.random.default_rng(seed(identity, 'real'))
    i = np.random.default_rng(seed(identity, 'imag'))
    out = np.empty(count, dtype=np.complex64)
    for first in range(0, count, 1048576):
        last = min(count, first + 1048576)
        out[first:last] = (r.standard_normal(last - first, dtype=np.float32) + 1j * i.standard_normal(last - first, dtype=np.float32)) / np.sqrt(2)
    return out
