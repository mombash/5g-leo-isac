"""Committed campaign behavior; source mapping in _config/extraction.json."""

def warm_backends(modules, specs, backends, reference, observation, alignment):
    import numpy as np
    import torch
    stream = modules['stripped_streaming_2026_10_26']
    stream.torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    c = torch.zeros((50 * stream.PER, 1792), dtype=torch.complex64, device='cuda')
    v = torch.zeros((200, 1792), device='cuda')
    for first in range(0, 50, 10):
        rx = torch.from_numpy(reference[first * stream.COUNT:stream.GUARD + (first + 10) * stream.COUNT].copy()).cuda()
        obs = torch.from_numpy(observation[first * stream.COUNT:(first + 10) * stream.COUNT].copy()).cuda()
        (clean, _) = stream.cancel_sinc(rx, obs, alignment[first:first + 10])
        c[first * stream.PER:(first + 10) * stream.PER] = stream.correlate(rx, clean)
        v[first * 4:(first + 10) * 4] = stream.variances5(rx, 10)
    for (name, backend) in backends.items():
        if backend['kind'] == 'kt':
            backend['backend'].stage(c, v, 0, 0, 53)
        elif backend['kind'] == 'ts':
            detector = modules['literature_two_step_2026_10_27_source_verified']
            (first, second, refine) = detector.gpu_callbacks(c, v)
            policy = backend['policy']
            result = first(policy.first_s, policy.lower, float('inf'))
            if policy.second_s and len(result['alert_parameters']):
                second(policy.first_s, policy.second_s, detector.propagate(result['alert_parameters'], policy.second_s))
            refine(0.0, policy.first_s, np.asarray(result['best_row']))
        else:
            method = specs[name]
            schedule = [(6, 27)] if backend['kind'] == 'simple' or method.get('noearly') else [(2, method['budgets'][0]), (4, method['budgets'][1]), (6, 27)]
            for (j, k) in schedule:
                backend['backend'].stage(c, v, 0, j, k)
    native = next((backend['backend'] for backend in backends.values() if backend['kind'] == 'query'))
    for slot in range(2):
        with torch.cuda.stream(stream.stream_resources()['search'][slot]):
            native.stage(c, v, 0, 6, 27)
    torch.cuda.synchronize()
