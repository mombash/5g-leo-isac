"""Simulate and run passive UAV sensing with 5G NR LEO downlinks."""
import argparse
import hashlib
import json
from pathlib import Path
from .assets import Assets, configuration
from .method_names import public_name


def method_name(value):
    try:
        return public_name(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    methods = commands.add_parser('methods', help='List receiver methods and fixed settings.')
    methods.add_argument('--details', action='store_true', help='Include file hashes and numerical configuration.')
    commands.add_parser('example', help='Check alert scoring on 40 small stored examples (CPU).')
    results = commands.add_parser('results', aliases=['reproduce'], help='Reaggregate stored detection and timing results (CPU).')
    results.add_argument('--no-bootstrap', action='store_true', help='Skip recomputing confidence intervals.')
    commands.add_parser('cohorts', help='List the scene/seed cohorts available for export.')
    inputs = commands.add_parser('export-inputs', aliases=['submission-inputs'], help='Export scene/seed metadata for explicit sample generation.')
    inputs.add_argument('--cohort', default='example', help='example, quality-part1–4, no-uav or calibration (see cohorts).')
    inputs.add_argument('--output', type=Path, required=True, help='New input directory.')
    inputs.add_argument('--waveform-root', type=Path, required=True, help='Directory for external or generated .c64 waveforms.')
    verify = commands.add_parser('verify-assets', help='Verify all model, policy and configuration files (CPU).')
    verify.add_argument('--assets', type=Path, required=True)
    prepare = commands.add_parser('prepared', help='Run one precomputed correlation attempt (GPU, no latency measurement).')
    prepare.add_argument('--assets', type=Path, required=True)
    prepare.add_argument('--method', type=method_name, required=True, metavar='METHOD', help='Public name from methods; compatibility identifiers are also accepted.')
    prepare.add_argument('--correlation', type=Path, required=True, help='(20000,1792) complex64 .npy array.')
    prepare.add_argument('--variance', type=Path, required=True, help='(200,1792) float32 .npy array.')
    prepare.add_argument('--output', type=Path, required=True, help='New JSON file.')
    geometry = commands.add_parser('geometry', help='Compute geometry and truth for a supplied scene (CPU).')
    for key in ['assets', 'scene', 'output']:
        geometry.add_argument('--' + key, type=Path, required=True)
    geometry_inputs = commands.add_parser('prepare-geometry', help='Compute geometry for the first exported scene (CPU).')
    for key in ['assets', 'inputs', 'output']:
        geometry_inputs.add_argument('--' + key, type=Path, required=True)
    paced = commands.add_parser('paced', help='Run sample-paced processing of raw reference/observation arrays (GPU).')
    paced.add_argument('--assets', type=Path, required=True)
    paced.add_argument('--method', type=method_name, required=True, metavar='METHOD', help='Public name from methods; compatibility identifiers are also accepted.')
    for key in ['reference', 'observation', 'alignment', 'output']:
        paced.add_argument('--' + key, type=Path, required=True)
    for key in ['reference', 'observation', 'alignment']:
        paced.add_argument('--warmup-' + key, type=Path, help='Optional disjoint calibration array; supply all three together.')
    sample = commands.add_parser('simulation-example', help='Compatibility command to export one scene and its seeds.')
    sample.add_argument('--output', type=Path, required=True)
    sample.add_argument('--waveform', type=Path, required=True)
    plan = commands.add_parser('generation-plan', help='Report raw-array size for an exported manifest (CPU).')
    plan.add_argument('--manifest', type=Path, required=True)
    commands.add_parser('matlab-path', help='Print the installed MATLAB helper directory.')
    build = commands.add_parser('build-interpolator', help='Build the sample-generation C++ interpolator with g++.')
    build.add_argument('--output', type=Path, required=True, help='New shared-library path, normally kernels.so.')
    generate = commands.add_parser('generate', help='Synthesize one simulation recording or a short diagnostic slice (CPU/C++).')
    for key in ['assets', 'manifest-directory', 'geometry-root', 'contract', 'scratch-root', 'output']:
        generate.add_argument('--' + key, type=Path, required=True)
    generate.add_argument('--split', help='Defaults to the exported inputs.json split.')
    generate.add_argument('--episode-id', help='Defaults to the first exported recording.')
    generate.add_argument('--duration', type=int, choices=[1, 4], help='Defaults to the exported recording duration.')
    generate.add_argument('--smoke-frames', type=int, choices=range(1, 6), help='Generate only 1–5 twenty-ms blocks, too short for receiver inference.')
    args = parser.parse_args()
    if args.command == 'methods':
        cfg = configuration()
        summary={group:{name:settings['label'] for name,settings in cfg[group].items()} for group in ['methods','controls']}
        print(json.dumps(cfg if args.details else summary, indent=2))
        return
    if args.command == 'example':
        from .examples import scoring_check
        print(json.dumps(scoring_check()))
        return
    if args.command in {'results', 'reproduce'}:
        from .reproduce import outcomes
        print(json.dumps(outcomes(bootstrap=not args.no_bootstrap), indent=2))
        return
    if args.command == 'cohorts':
        from .reproduce import cohorts
        print(json.dumps(cohorts(), indent=2))
        return
    if args.command in {'export-inputs', 'submission-inputs'}:
        if args.cohort == 'example':
            from .simulation import write_example
            result = write_example(args.output, args.waveform_root / 'example.c64')
        else:
            from .reproduce import export_inputs
            result = export_inputs(args.output, args.cohort, waveform_root=args.waveform_root)
        print(json.dumps(result, indent=2))
        return
    if args.command == 'simulation-example':
        from .simulation import write_example
        print(json.dumps(write_example(args.output, args.waveform)))
        return
    if args.command == 'generation-plan':
        from .simulation import generation_plan
        print(json.dumps(generation_plan(args.manifest), indent=2))
        return
    if args.command == 'matlab-path':
        from importlib.resources import files
        print(str(files('leo_ofdm_repro').joinpath('matlab')))
        return
    if args.command == 'build-interpolator':
        from importlib.resources import files
        import subprocess
        if args.output.exists():
            raise FileExistsError('Existing shared library is preserved.')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['g++', '-O3', '-fPIC', '-shared', '-fopenmp',
                        str(files('leo_ofdm_repro').joinpath('native/kernels.cpp')),
                        '-o', str(args.output)], check=True)
        print(str(args.output.resolve()))
        return
    assets = Assets(args.assets)
    if args.command == 'verify-assets':
        print(json.dumps(dict(passed=True, files=len(assets.manifest['files']))))
        return
    if args.command == 'prepare-geometry':
        from .simulation import prepare_geometry
        print(json.dumps(prepare_geometry(assets, args.inputs, args.output), indent=2))
        return
    if args.command == 'generate':
        import numpy as np
        from .simulation import recording_generator, recording_slice
        if args.output.exists():
            raise FileExistsError('Existing raw output directory is preserved.')
        metadata_path = args.manifest_directory / 'inputs.json'
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        split = args.split or metadata.get('split')
        episode_id = args.episode_id or metadata.get('episode_id')
        if split is None or episode_id is None:
            parser.error('Supply --split and --episode-id, or use a directory from export-inputs.')
        manifest = json.loads((args.manifest_directory / (split + '_manifest.json')).read_text())
        duration = args.duration if args.duration is not None else manifest.get('duration_s', 4)
        row = next((row for row in manifest['episodes'] if row['episode_id'] == episode_id), None)
        if row is None:
            raise ValueError('Recording ID is absent from the supplied manifest.')
        kwargs = dict(manifest_directory=args.manifest_directory, geometry_root=args.geometry_root,
                      contract=args.contract, scratch_root=args.scratch_root, split=split)
        values = recording_slice(**kwargs, row=row, frames=args.smoke_frames) if args.smoke_frames else recording_generator(**kwargs, duration_s=duration).recording(row)
        args.output.mkdir(parents=True)
        for key, value in zip(['reference', 'observation', 'alignment'], values):
            np.save(args.output / (key + '.npy'), value)
        result = dict(episode_id=row['episode_id'], duration_s=args.smoke_frames * .02 if args.smoke_frames else duration,
                      sliced_smoke=bool(args.smoke_frames), array_shapes=[list(v.shape) for v in values],
                      array_dtypes=[str(v.dtype) for v in values])
        (args.output / 'generation.json').write_text(json.dumps(result, indent=2) + '\n')
        print(str(args.output))
        return
    if args.output.exists():
        raise FileExistsError('Existing outputs are preserved; select a new output path.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.command == 'geometry':
        import numpy as np
        from .geometry import geometry
        cfg = json.loads((assets.root / 'physical_configuration.json').read_text())
        arrays, summary = geometry(cfg, json.loads(args.scene.read_text()))
        if args.output.suffix != '.npz':
            raise ValueError('Geometry output must use .npz.')
        if args.output.with_suffix('.json').exists():
            raise FileExistsError('Geometry summary already exists.')
        np.savez_compressed(args.output, **arrays)
        args.output.with_suffix('.json').write_text(json.dumps(summary, indent=2) + '\n')
    elif args.command == 'paced':
        import numpy as np
        import torch
        from .runtime import Runtime
        from .paced import validate_inputs
        if not torch.cuda.is_available():
            raise RuntimeError('Receiver inference requires CUDA.')
        inputs = {k: np.load(getattr(args, k), allow_pickle=False) for k in ['reference', 'observation', 'alignment']}
        validate_inputs(**inputs)
        hashes = {k: hashlib.sha256(memoryview(a)).hexdigest() for k, a in inputs.items()}
        warm_paths = [getattr(args, 'warmup_' + k) for k in ['reference', 'observation', 'alignment']]
        if any(p is not None for p in warm_paths) and not all(p is not None for p in warm_paths):
            raise ValueError('Supply all three warmup arrays together.')
        warmup = None if not all(p is not None for p in warm_paths) else tuple(np.load(p, allow_pickle=False) for p in warm_paths)
        result = Runtime(args.assets).paced(args.method, **inputs, warmup_inputs=warmup)
        result['input_array_sha256'] = hashes
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    else:
        import numpy as np
        import torch
        from .runtime import Runtime
        if not torch.cuda.is_available():
            raise RuntimeError('Receiver inference requires CUDA.')
        c = np.load(args.correlation, mmap_mode='r')
        v = np.load(args.variance, mmap_mode='r')
        if c.shape != (20000, 1792) or v.shape != (200, 1792):
            raise ValueError('Supply exactly one second of prepared correlations and variances.')
        hashes = {name: hashlib.sha256(memoryview(a)).hexdigest() for name, a in [('correlation', c), ('variance', v)]}
        result = Runtime(args.assets).prepared(args.method, torch.from_numpy(c.copy()).cuda(), torch.from_numpy(v.copy()).cuda())
        result['input_array_sha256'] = hashes
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(str(args.output))


if __name__ == '__main__':
    main()
