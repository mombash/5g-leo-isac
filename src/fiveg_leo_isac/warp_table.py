"""Build interpolation tables for delay-migration correction."""
import numpy as np

def interpolation_table(taps=48, phases=4096):
    offsets = np.arange(taps) - (taps // 2 - 1)
    fractions = np.arange(phases + 1) / phases
    table = np.sinc(fractions[:, None] - offsets) * np.kaiser(taps, 10)[None, :]
    table /= table.sum(axis=1, keepdims=True)
    return np.asarray(table, dtype=np.float32)
