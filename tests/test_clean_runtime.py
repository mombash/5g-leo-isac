"""Alert-scoring and asset-integrity contracts."""
import json
from importlib.resources import files
from pathlib import Path
import shutil
import tempfile
import unittest
from leo_ofdm_repro.scoring import score_emissions,standardized_metrics,InvalidEvidence
from leo_ofdm_repro.numerics import float32_up
from leo_ofdm_repro.examples import scoring_check
from leo_ofdm_repro.assets import Assets,configuration


class Scoring(unittest.TestCase):
    def event(self,t,correct=True,alert=True):
        return dict(observation_s=t-.1,available_s=t,alert=alert,prediction=[30.,0. if correct else 100.,0.])
    def score(self,events,present=True):
        return score_emissions(events,present=present,truth_at=lambda t:[0.,0.],terminal_s=1.4)
    def test_later_correct_does_not_erase_wrong_first(self):
        r=self.score([self.event(.7,False),self.event(.9,True)])
        self.assertTrue(r['wrong_first']);self.assertFalse(r['miss']);self.assertTrue(r['final_correct'])
        self.assertEqual(r['first_correct_available_s'],.9)
    def test_first_correct_survives_wrong_final(self):
        r=self.score([self.event(.7,True),self.event(1.1,False)])
        self.assertFalse(r['miss']);self.assertFalse(r['final_correct']);self.assertFalse(r['wrong_first'])
        self.assertEqual(r['operational_s'],.7)
    def test_first_positive_is_distinct_from_first_check(self):
        r=self.score([self.event(.7,True,False),self.event(.9,False),self.event(1.1,True)])
        self.assertTrue(r['wrong_first']);self.assertEqual(r['first_correct_available_s'],1.1)
    def test_any_empty_false_alert_survives_negative_final(self):
        r=self.score([self.event(.7),self.event(1.1,alert=False)],present=False)
        self.assertTrue(r['any_false_alert']);self.assertFalse(r['final_false_alert']);self.assertEqual(r['operational_s'],1.4)
    def test_miss_retains_terminal_time(self):
        r=self.score([self.event(.7,False)])
        self.assertTrue(r['miss']);self.assertIsNone(r['first_correct_available_s']);self.assertEqual(r['operational_s'],1.4)
    def test_events_cannot_precede_available_samples(self):
        e=self.event(.7);e['observation_s']=.8
        with self.assertRaises(InvalidEvidence):self.score([e])
    def test_duplicate_event_indices_rejected(self):
        a=self.event(.7);b=self.event(.9);a['event_index']=b['event_index']=0
        with self.assertRaises(InvalidEvidence):self.score([a,b])
    def test_mixture_weight_is_not_inferred_from_counts(self):
        t=self.score([self.event(.7)]);n=self.score([self.event(1.1,alert=False)],False)
        r=standardized_metrics([t,t,t,n],target_weight=.5)
        self.assertAlmostEqual(r['standardized_operational_mean_s'],1.05)
    def test_actual_float32_threshold_rounding(self):
        x=configuration()['methods']['two-step-early']['policy']['upper'];y=float32_up(x)
        self.assertGreaterEqual(y,x);self.assertEqual(float32_up(y),y)
        self.assertEqual(y,24.485971450805664)
    def test_saved_scoring_examples(self):
        r=scoring_check();self.assertTrue(r['passed']);self.assertEqual(r['saved_attempt_examples'],40)


class SelectedAssets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=Path(__file__).resolve().parents[1];cls.real=cls.root/'selected_assets'
        if not cls.real.exists():raise unittest.SkipTest('Run asset checks from a checkout containing selected_assets.')
        cls.tmp=Path(tempfile.gettempdir())
    def test_genuine_assets(self):
        self.assertEqual(len(Assets(self.real).manifest['files']),7)
    def test_empty_manifest_not_a_valid_selected_bundle(self):
        with tempfile.TemporaryDirectory(dir=self.tmp) as name:
            p=Path(name);d=json.loads((self.real/'manifest.json').read_text());d['files']=[]
            (p/'manifest.json').write_text(json.dumps(d))
            with self.assertRaises(RuntimeError):Assets(p)
    def test_tampered_prior_and_rewritten_manifest_rejected(self):
        import hashlib
        with tempfile.TemporaryDirectory(dir=self.tmp) as name:
            p=Path(name);shutil.copytree(self.real,p,dirs_exist_ok=True)
            prior=p/'policies/phase_refinement/rate_prior.npy';raw=bytearray(prior.read_bytes());raw[-1]^=1;prior.write_bytes(raw)
            d=json.loads((p/'manifest.json').read_text())
            for r in d['files']:
                if r['path'].endswith('rate_prior.npy'):r['sha256']=hashlib.sha256(raw).hexdigest()
            (p/'manifest.json').write_text(json.dumps(d))
            with self.assertRaises(RuntimeError):Assets(p)


if __name__=='__main__':unittest.main()
