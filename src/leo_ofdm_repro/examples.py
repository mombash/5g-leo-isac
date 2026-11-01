"""Small installed-package examples with no external source/data dependency."""
import json
from importlib.resources import files
from .scoring import score_emissions


def scoring_check():
    fixture=json.loads(files('leo_ofdm_repro').joinpath('_config/scoring_examples.json').read_text())
    for case in fixture['cases']:
        actual=score_emissions(case['events'],present=case['present'],
                              truth_at=lambda t:case['truth_by_horizon'][str(t)],terminal_s=case['terminal_s'])
        for name,expected in case['expected'].items():
            if actual[name]!=expected:raise AssertionError(f'{case["name"]}: {name} differs')
    return dict(passed=True,saved_attempt_examples=len(fixture['cases']),new_inference=False)
