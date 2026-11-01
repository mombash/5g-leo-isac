"""Concrete dependencies used by the selected candidate-query stage."""
from types import SimpleNamespace
from .parameters import TIMES,PER
from .coherent import CoherentEpisode
from .verifier import LocalVerifier
from .native import native_rows
from .refinement import classical_refine,phase_traces
from .evidence import Evidence

runtime=SimpleNamespace(Evidence=Evidence)
