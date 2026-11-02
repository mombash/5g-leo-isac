"""Run receiver processing as host-resident digital samples become available."""
import ast
import concurrent.futures as cf
import inspect
import numpy as np
import torch
from .context import use
from .parameters import FS,COUNT,GUARD,TIMES
from .duration_adapters import ShortScheduler
from .registry import ADAPTIVE_METHODS,validate_name


def validate_inputs(reference,observation,alignment):
    if reference.dtype!=np.complex64 or observation.dtype!=np.complex64:
        raise ValueError('Raw reference and observation must be complex64.')
    if reference.ndim!=1 or observation.ndim!=1:
        raise ValueError('Raw inputs must be one-dimensional.')
    duration=len(observation)/FS
    if duration not in (1.,4.):raise ValueError('Supply exactly one or four seconds of raw samples.')
    if len(reference)!=len(observation)+GUARD:
        raise ValueError('Reference must include the preceding 10 ms guard.')
    if alignment.shape!=(int(duration*50),2) or not np.issubdtype(alignment.dtype,np.floating):
        raise ValueError('Alignment must have one [delay_samples,delay_rate_samples_per_s] row per 20 ms.')
    if not np.isfinite(alignment).all():raise ValueError('Nonfinite alignment.')
    return int(duration)


def short_class(cls):
    """Resize buffers and controllers for one detection attempt."""
    node=ast.parse(inspect.getsource(cls)).body[0]
    node=ShortScheduler().visit(node)
    scope=dict(inspect.getmodule(cls).__dict__)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),
                 cls.__module__+'#one-second-adapter','exec'),scope)
    return scope[cls.__name__]


def make_record(runtime,name,reference,observation,alignment):
    from . import scheduler,hybrid_scheduler
    from .adaptive_scheduler import AdaptiveRecord
    from .two_step_scheduler import StreamingRecord as TwoStepRecord
    from .two_step_backend import Backend as TwoStepBackend
    from .two_step import Policy
    from .numerics import float32_up
    duration=validate_inputs(reference,observation,alignment)
    settings=runtime.settings(name);cls=hybrid_scheduler.StreamingRecord if name=='hybrid-integration' else scheduler.StreamingRecord
    if duration==1:cls=short_class(cls)
    policy=None
    if name in ADAPTIVE_METHODS:
        cls=type('AdaptiveAttemptRecord',(cls,),{'controller':AdaptiveRecord.controller})
        backend=runtime.backend(name);backend.force_full={}
    elif name in {'two-step-fixed','two-step-early'}:
        cls=type('SequentialAttemptRecord',(cls,),{'controller':TwoStepRecord.controller})
        p=dict(settings['policy'])
        if settings.get('round_bounds_up'):p['lower']=float32_up(p['lower']);p['upper']=float32_up(p['upper'])
        policy=Policy(**p);backend=TwoStepBackend(policy)
        settings=dict(id=name,kind='two_step',policy={})
    elif name=='dual-scale':
        from .dual_scale_backend import StreamingBackend
        backend=StreamingBackend()
    elif name=='hybrid-integration':
        from .hybrid_backend import Backend
        settings.update(kind='simple',policy=dict(looks=[0],k=53))
        backend=Backend()
    elif name=='exhaustive-classical':backend=runtime.backend(name)
    else:raise ValueError('Unknown receiver method.')
    record=cls(settings,backend,{},reference,observation,alignment,'pipeline')
    if policy is not None:record.policy=policy
    if name=='hybrid-integration':
        record.pool.shutdown();record.pool=cf.ThreadPoolExecutor(max_workers=1);record.slots=[0]
    return record


def verify_causality(output,name):
    for stage in output['shared_stages']:
        if stage['start_s']+1e-6<stage['ready_s']:raise RuntimeError('Frontend used future samples.')
    for stage in output['executed_searches']:
        horizon=1. if name=='hybrid-integration' else float(TIMES[stage['j']])
        if stage['start_s']+1e-6<stage['ordinal']+horizon:raise RuntimeError('Search used future samples.')
        wanted=stage['ordinal']*50+round(horizon/.02)
        if not any(p['first_frame']<wanted<=p['last_frame'] and p['finish_s']<=stage['start_s']+1e-6
                   for p in output['shared_stages']):raise RuntimeError('Search preceded required preparation.')
    from .scoring import events_from_attempt
    for attempt in output['attempts']:
        for event in events_from_attempt(attempt):
            if event['available_s']+1e-6<event['observation_s']:raise RuntimeError('Output preceded observation.')


@torch.no_grad()
def run(runtime,name,reference,observation,alignment,*,warmup_inputs=None):
    duration=validate_inputs(reference,observation,alignment)
    warm_reference,warm_observation,warm_alignment=(reference,observation,alignment) if warmup_inputs is None else warmup_inputs
    validate_inputs(warm_reference,warm_observation,warm_alignment)
    with use(runtime.assets):
        record=make_record(runtime,name,reference,observation,alignment)
        # Warm all physical shapes and both persistent search streams before timing.
        from .warmup import warm_backends
        from . import scheduler,two_step
        kind='adaptive' if name in ADAPTIVE_METHODS else 'two-step' if name in {'two-step-fixed','two-step-early'} else 'hybrid-integration' if name=='hybrid-integration' else 'simple'
        backends={name:dict(kind=kind,backend=record.b)};specs={name:runtime.settings(name)}
        if kind=='two-step':backends[name]['policy']=record.policy
        if kind!='adaptive':
            backends['adaptive-neural']=dict(kind='adaptive',backend=runtime.backend('adaptive-neural'));specs['adaptive-neural']=runtime.settings('adaptive-neural')
        try:
            warm_backends({'streaming':scheduler,
                'two_step':two_step},specs,backends,warm_reference,warm_observation,warm_alignment)
        except BaseException as error:
            record.fail(error);record.pool.shutdown(wait=True)
            raise
        if kind=='adaptive':record.b.force_full={}
        torch.cuda.synchronize()
        try:output=record.run()
        except BaseException as error:
            record.fail(error);record.pool.shutdown(wait=True)
            raise
    if name=='hybrid-integration':output['max_search_workers']=1
    verify_causality(output,name)
    output.update(schema=2,method=name,duration_s=duration,mode='pipeline',
        execution='sample-paced-receiver',timing_valid=True,causality_verified=True,
        timing_boundary='Host-resident digital sample-paced input; initialization, disk/RF/ADC/acquisition/alignment estimation excluded.',
        source_commits=runtime.assets.configuration['source_commits'],
        operating_point=runtime.assets.configuration['operating_point'],
        one_second_adapter=duration==1,
        warmup_applied=True,warmup_input='First second of explicit warmup inputs' if warmup_inputs is not None else 'First second of receiver input; supply disjoint calibration inputs for the paper warmup protocol',
        model_provenance=runtime.assets.configuration['model_provenance'],
        warmup_sequence='Physical stage shapes followed by both persistent search streams; neural search backend also warmed for baseline runs.',
        timing_interpretation='Actual timestamps for this execution; compare performance only under a matched host, warmup, inputs and load.')
    return output,record
