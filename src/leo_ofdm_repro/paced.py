"""Original digital sample-paced receiver, with explicit portable raw inputs."""
import ast
import concurrent.futures as cf
import inspect
import numpy as np
import torch
from .context import use
from .parameters import FS,COUNT,GUARD,TIMES
from .duration_adapters import ShortScheduler


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
    """Apply the committed one-second capacity/controller-count adapter."""
    node=ast.parse(inspect.getsource(cls)).body[0]
    node=ShortScheduler().visit(node)
    scope=dict(inspect.getmodule(cls).__dict__)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),
                 cls.__module__+'#committed-one-second-adapter','exec'),scope)
    return scope[cls.__name__]


def make_record(runtime,name,reference,observation,alignment):
    from . import scheduler,kt_scheduler
    from .query_scheduler import QueryRecord
    from .ts_scheduler import StreamingRecord as TwoStepRecord
    from .ts_backend import Backend as TwoStepBackend
    from .two_look import Policy
    from .numerics import float32_up
    duration=validate_inputs(reference,observation,alignment)
    settings=runtime.settings(name);cls=kt_scheduler.StreamingRecord if name=='KT+LVD' else scheduler.StreamingRecord
    if duration==1:cls=short_class(cls)
    policy=None
    if name in {'N5','S5','C9'}:
        cls=type('SelectedQueryRecord',(cls,),{'controller':QueryRecord.controller})
        backend=runtime.backend(name);backend.force_full={}
    elif name in {'TS1','TS2'}:
        cls=type('SelectedTwoStepRecord',(cls,),{'controller':TwoStepRecord.controller})
        p=dict(settings['policy'])
        if name=='TS2':p['lower']=float32_up(p['lower']);p['upper']=float32_up(p['upper'])
        policy=Policy(**p);backend=TwoStepBackend(policy)
        settings=dict(id=name,kind='two_step',policy={})
    elif name=='DS':
        from .ds_backend import StreamingBackend
        backend=StreamingBackend()
    elif name=='KT+LVD':
        from .kt_backend import Backend
        settings.update(kind='simple',policy=dict(looks=[0],k=53))
        backend=Backend()
    elif name=='E27':backend=runtime.backend(name)
    else:raise ValueError('Unknown selected method.')
    record=cls(settings,backend,{},reference,observation,alignment,'pipeline')
    if policy is not None:record.policy=policy
    if name=='KT+LVD':
        record.pool.shutdown();record.pool=cf.ThreadPoolExecutor(max_workers=1);record.slots=[0]
    return record


def verify_causality(output,name):
    for stage in output['shared_stages']:
        if stage['start_s']+1e-6<stage['ready_s']:raise RuntimeError('Frontend used future samples.')
    for stage in output['executed_searches']:
        horizon=1. if name=='KT+LVD' else float(TIMES[stage['j']])
        if stage['start_s']+1e-6<stage['ordinal']+horizon:raise RuntimeError('Search used future samples.')
        wanted=stage['ordinal']*50+round(horizon/.02)
        if not any(p['first_frame']<wanted<=p['last_frame'] and p['finish_s']<=stage['start_s']+1e-6
                   for p in output['shared_stages']):raise RuntimeError('Search preceded required preparation.')
    from .scoring import events_from_attempt
    for attempt in output['attempts']:
        for event in events_from_attempt(attempt):
            if event['available_s']+1e-6<event['observation_s']:raise RuntimeError('Output preceded observation.')


@torch.no_grad()
def run(runtime,name,reference,observation,alignment):
    duration=validate_inputs(reference,observation,alignment)
    with use(runtime.assets):
        record=make_record(runtime,name,reference,observation,alignment)
        # Same committed warmup sequence, outside the measured scheduler origin.
        # Supply its native-query stream warmup even for a single baseline run.
        from .warmup import warm_backends
        from . import scheduler,two_look
        kind='query' if name in {'N5','S5','C9'} else 'ts' if name in {'TS1','TS2'} else 'kt' if name=='KT+LVD' else 'simple'
        backends={name:dict(kind=kind,backend=record.b)};specs={name:runtime.settings(name)}
        if kind=='ts':backends[name]['policy']=record.policy
        if kind!='query':
            backends['N5']=dict(kind='query',backend=runtime.backend('N5'));specs['N5']=runtime.settings('N5')
        try:
            warm_backends({'stripped_streaming_2026_10_26':scheduler,
                'literature_two_step_2026_10_27_source_verified':two_look},specs,backends,reference,observation,alignment)
        except BaseException as error:
            record.fail(error);record.pool.shutdown(wait=True)
            raise
        if kind=='query':record.b.force_full={}
        torch.cuda.synchronize()
        try:output=record.run()
        except BaseException as error:
            record.fail(error);record.pool.shutdown(wait=True)
            raise
    if name=='KT+LVD':output['max_search_workers']=1
    verify_causality(output,name)
    output.update(schema=2,method=name,duration_s=duration,mode='pipeline',
        execution='combined-quality-and-timing',timing_valid=True,causality_verified=True,
        timing_boundary='Host-resident digital sample-paced input; initialization, disk/RF/ADC/acquisition/alignment estimation excluded.',
        source_commits=runtime.assets.configuration['source_commits'],
        operating_point=runtime.assets.configuration['operating_point'],
        one_second_adapter=duration==1,
        warmup_applied=True,warmup_input='First second of the supplied raw recording; no truth or parameter adaptation.',
        warmup_sequence='Committed campaign warm_backends; selected route plus N5 native-search-stream warmup when required.',
        timing_interpretation='Actual timestamps for this execution; compare performance only under a matched host, warmup, inputs and load.')
    return output,record
