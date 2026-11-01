"""Selected receiver execution from portable, hash-verified assets."""
import copy
import numpy as np
import torch
from .assets import Assets
from .context import use
from .parameters import TIMES

METHODS=('N5','S5','C9','E27','DS','TS1','TS2','KT+LVD')


class Runtime:
    def __init__(self,asset_root):
        self.assets=Assets(asset_root)
        self.backends={}
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False

    def settings(self,name):return copy.deepcopy(self.assets.configuration['methods'][name])

    def paced(self,name,reference,observation,alignment):
        from .paced import run
        return run(self,name,reference,observation,alignment)[0]

    def backend(self,name):
        if name not in METHODS:raise ValueError('Unknown selected method.')
        if name not in self.backends:
            with use(self.assets):
                if name in {'N5','S5','C9'}:
                    from .query_stage import Backend
                    self.backends[name]=Backend(self.settings(name))
                elif name=='E27':
                    from .base import BaseBackend
                    self.backends[name]=BaseBackend(self.settings(name))
        return self.backends.get(name)

    @torch.no_grad()
    def prepared(self,name,correlation,variance):
        if correlation.shape!=(20000,1792) or variance.shape!=(200,1792):raise ValueError('Expected exactly one 1 s prepared attempt.')
        if correlation.dtype!=torch.complex64 or variance.dtype!=torch.float32:raise ValueError('Preserve complex64/float32 inputs.')
        if correlation.device.type!='cuda' or variance.device.type!='cuda':raise ValueError('Selected inference requires CUDA.')
        settings=self.settings(name)
        with use(self.assets):
            if name in {'N5','S5','C9'}:
                backend=self.backend(name);events=[];prior=False
                for j,k in [(2,settings['budgets'][0]),(4,settings['budgets'][1]),(6,27)]:
                    if j==6:backend.force_full[0]=prior
                    value=backend.stage(correlation,variance,0,j,k)
                    point=value['point'];alert=bool(point[0]>settings['threshold']);prior|=alert
                    events.append(dict(observation_s=float(TIMES[j]),prediction=point.tolist(),alert=alert,
                                       selected_ids=value.get('selected_ids'),executed_rates=value.get('executed_rates',k),
                                       final_gate_rejected=value.get('final_gate_rejected',False)))
            elif name=='E27':
                value=self.backend(name).stage(correlation,variance,0,6,27)
                events=[dict(observation_s=1.,prediction=value['point'].tolist(),alert=bool(value['point'][0]>settings['threshold']))]
            elif name=='DS':
                from .dual_scale import confirm_rows
                value=confirm_rows(correlation,variance,1.,batch_rates=2)
                events=[dict(observation_s=1.,prediction=value['point'].tolist(),alert=bool(value['point'][0]>settings['threshold']))]
            elif name in {'TS1','TS2'}:
                from .two_look import Policy
                from .ts_backend import Backend
                from .numerics import float32_up
                policy=dict(settings['policy'])
                if name=='TS2':
                    policy['lower']=float32_up(policy['lower']);policy['upper']=float32_up(policy['upper'])
                backend=Backend(Policy(**policy));j=int(np.argmin(abs(TIMES-policy['first_s'])))
                while True:
                    value=backend.stage(correlation,variance,0,j,27)
                    if value['terminal']:break
                    j=6
                point=np.nan_to_num(value['point'],nan=-1e30,neginf=-1e30,posinf=-1e30)
                events=[dict(observation_s=float(TIMES[j]),prediction=point.tolist(),alert=value['alert'],gate_score=value['gate_score'])]
            elif name=='KT+LVD':
                from .kt_lvd import infer
                value=infer(correlation,variance,nt=4000,batch=32,time_transform='matrix',lag_block=256)
                events=[dict(observation_s=1.,prediction=value['point'],alert=bool(value['point'][0]>settings['threshold']))]
            else:raise ValueError('Unknown selected method.')
        return dict(method=name,events=events,any_alert=any(e['alert'] for e in events),final_alert=events[-1]['alert'],
                    timing_valid=False,source_commits=self.assets.configuration['source_commits'],
                    operating_point=self.assets.configuration['operating_point'])
