"""Integrity of recorded cohort relationships and canonical receiver names."""
import json
import unittest
from importlib.resources import files
from fiveg_leo_isac.assets import configuration
from fiveg_leo_isac.registry import METHODS, CONTROLS
from fiveg_leo_isac.results import resource


class RecordedInputs(unittest.TestCase):
    def test_result_methods_match_registry(self):
        data = resource('recorded_results')
        self.assertEqual(data['methods'], list(METHODS))
        for recording in data['quality']:
            self.assertEqual(set(recording['methods']), set(METHODS))
        for recording in data['timing']:
            self.assertIn(recording['method'], METHODS + CONTROLS)

    def test_cohort_references_and_explicit_noise_seeds(self):
        for value in resource('cohort_inputs').values():
            manifest = value['manifest']
            scenes = {s['scene_id']: s for s in manifest['scenes']}
            self.assertEqual(len(scenes), len(manifest['scenes']))
            self.assertEqual(len({r['episode_id'] for r in manifest['episodes']}), len(manifest['episodes']))
            for row in manifest['episodes']:
                self.assertIn(row['scene_id'], scenes)
                self.assertEqual(row['split'], manifest['split'])
                self.assertIs(type(row['noise_id']), int)
                self.assertIs(type(row['reference_noise_id']), int)
                scene = scenes[row['scene_id']]
                self.assertEqual(scene['split'], manifest['split'])
                wave = scene['waveform_id']
                self.assertLess(wave, len(manifest['waveforms']))
                if 'waveform_id' in row:
                    self.assertEqual(row['waveform_id'], wave)

    def test_example_preserves_its_cohort_scene_and_recording(self):
        example = json.loads(files('fiveg_leo_isac').joinpath('_config/simulation_example.json').read_text())['manifest']
        cohort = resource('cohort_inputs')['quality-part2']['manifest']
        self.assertEqual(example['scenes'][0], cohort['scenes'][0])
        self.assertEqual(example['episodes'][0], cohort['episodes'][0])
        for name, method in configuration()['methods'].items():
            self.assertEqual(method['id'], name)


if __name__ == '__main__':
    unittest.main()
