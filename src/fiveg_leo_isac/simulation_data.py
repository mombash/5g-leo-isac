"""Synthesize received reference, leakage, echo and noise samples."""
import json
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicSpline
from .parameters import FS,FC,C
from .noise import noise
from .interpolation import interpolate


class EpisodeData:

    def __init__(self, split):
        self.split = split
        self.manifest = json.loads((OUT / f'{split}_manifest.json').read_text())
        if self.manifest['sealed']:
            raise RuntimeError('This input manifest is sealed.')
        contract = json.loads((OUT / 'contract.json').read_text())
        if contract['reference_choice'].get('status') != 'approved':
            raise RuntimeError('The contract must specify an approved reference polarization.')
        self.scenes = {s['scene_id']: s for s in self.manifest['scenes']}
        self.last_scene = None
        self.clean = None
        self.last_pair = None
        self.received_reference = None
        self.surveillance_noise = None

    def physical(self, scene_id, component):
        scene = self.scenes[scene_id]
        summary = json.loads((OUT / 'geometry' / self.split / (scene_id + '.json')).read_text())
        cache = CACHE / 'clean' / self.split / scene_id / summary['geometry_fingerprint'][:16] / (component + '.npy')
        if cache.exists():
            return np.load(cache, mmap_mode='r')
        g = np.load(OUT / 'geometry' / self.split / (scene_id + '.npz'))
        t = g['time_s']
        ref = CubicSpline(t, g['reference_length_m'])
        base = float(ref(0))
        names = {'reference': ('reference_length_m', 'reference_field'), 'echo': ('echo_length_m', 'echo_field'), 'direct': ('direct_length_m', 'direct_field')}
        length = CubicSpline(t, g[names[component][0]])
        field = CubicSpline(t, g[names[component][1]])
        wave = self.manifest['waveforms'][scene['waveform_id']]
        tx = np.memmap(wave['cache_path'], mode='r', dtype='<c8')
        count = int(round((4.01 if component == 'reference' else 4) * FS))
        origin = -0.01 if component == 'reference' else 0.0
        if scene['null_scene']:
            output = np.empty(count, dtype=np.complex64)
        else:
            cache.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache.with_suffix('.tmp.npy')
            output = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.complex64, shape=(count,))
        for first in range(0, count, 1048576):
            last = min(count, first + 1048576)
            clock = origin + np.arange(first, last) / FS
            distance = length(clock) - base
            coordinates = (clock - distance / C + 0.01) * FS
            valid = (coordinates >= 32) & (coordinates < len(tx) - 33)
            samples = interpolate(tx, np.clip(coordinates, 32, len(tx) - 33))
            samples[~valid] = 0
            samples *= np.exp(-2j * np.pi * FC * distance / C).astype(np.complex64)
            if component == 'echo':
                samples *= np.exp(1j * scene['scattering_phase_rad'])
            output[first:last] = samples * (field(clock) / np.sqrt(FS)).astype(np.complex64)
        if hasattr(output, 'flush'):
            output.flush()
            temporary.replace(cache)
        return output

    def recording(self, row):
        scene_id = row['scene_id']
        if scene_id != self.last_scene:
            self.clean = {'reference': self.physical(scene_id, 'reference'), 'direct': self.physical(scene_id, 'direct')}
            if row['present']:
                self.clean['echo'] = self.physical(scene_id, 'echo')
            self.last_scene = scene_id
            self.last_pair = None
        pair = (scene_id, row['noise_id'], row['reference_noise_id'])
        if pair != self.last_pair:
            self.received_reference = np.asarray(self.clean['reference'] + noise(len(self.clean['reference']), row['reference_noise_id']), dtype=np.complex64)
            self.surveillance_noise = noise(int(4 * FS), row['noise_id'])
            self.last_pair = pair
        observation = np.asarray(self.clean['direct'] + self.surveillance_noise, dtype=np.complex64)
        if row['present']:
            observation += np.sqrt(row['rcs_m2'] / 0.03) * self.clean['echo']
        g = np.load(OUT / 'geometry' / self.split / (scene_id + '.npz'))
        t = g['time_s']
        ref_emission = t - g['reference_length_m'] / C
        inverse = CubicSpline(ref_emission, t)
        delta = (inverse(t - g['direct_length_m'] / C) - t) * FS
        spline = CubicSpline(t, delta)
        centres = (np.arange(200) + 0.5) * 0.02
        alignment = np.column_stack((spline(centres), spline(centres, 1)))
        return (self.received_reference, observation, alignment)

    def truth(self, row):
        if not row['present']:
            return None
        summary = json.loads((OUT / 'geometry' / self.split / (row['scene_id'] + '.json')).read_text())
        return summary['labels']
