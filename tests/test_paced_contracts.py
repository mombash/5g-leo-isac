"""Raw-input and measured-time contracts of the standalone paced interface."""
import unittest
import numpy as np
from leo_ofdm_repro.paced import validate_inputs,verify_causality


class PacedContracts(unittest.TestCase):
    def arrays(self,seconds=1):
        sample=np.zeros(1,np.complex64)
        ref=np.lib.stride_tricks.as_strided(sample,shape=(int(seconds*30.72e6)+307200,),strides=(0,))
        obs=np.lib.stride_tricks.as_strided(sample,shape=(int(seconds*30.72e6),),strides=(0,))
        return ref,obs,np.zeros((seconds*50,2),np.float64)
    def test_selected_one_and_four_second_forms(self):
        for seconds in [1,4]:self.assertEqual(validate_inputs(*self.arrays(seconds)),seconds)
    def test_missing_reference_guard_rejected(self):
        ref,obs,al=self.arrays()
        with self.assertRaises(ValueError):validate_inputs(ref[307200:],obs,al)
    def test_wrong_raw_dtype_rejected(self):
        ref,obs,al=self.arrays()
        with self.assertRaises(ValueError):validate_inputs(ref.view(np.float32),obs,al)
    def test_nonfinite_alignment_rejected(self):
        ref,obs,al=self.arrays();al[0,0]=np.nan
        with self.assertRaises(ValueError):validate_inputs(ref,obs,al)
    def test_search_cannot_precede_frontend_completion(self):
        out=dict(attempts=[],shared_stages=[dict(first_frame=0,last_frame=50,ready_s=1.,start_s=1.,finish_s=1.2)],
                 executed_searches=[dict(ordinal=0,j=6,start_s=1.1)])
        with self.assertRaises(RuntimeError):verify_causality(out,'E27')
    def test_frontend_cannot_precede_samples(self):
        out=dict(attempts=[],executed_searches=[],shared_stages=[dict(ready_s=.2,start_s=.19)])
        with self.assertRaises(RuntimeError):verify_causality(out,'N5')
    def test_actual_available_times_pass(self):
        out=dict(shared_stages=[dict(first_frame=0,last_frame=50,ready_s=1.,start_s=1.,finish_s=1.2)],
          executed_searches=[dict(ordinal=0,j=6,start_s=1.2)],attempts=[dict(start=0,decision_s=1.5,observation_s=1.,output=dict(alert=False,prediction=[0.,0.,0.]))])
        verify_causality(out,'E27')


if __name__=='__main__':unittest.main()
