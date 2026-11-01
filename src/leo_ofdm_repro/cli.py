"""Portable commands for the selected reproducible implementation."""
import argparse
import hashlib
import json
from pathlib import Path
from .assets import Assets,configuration


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('methods',help='Print selected retained parameters and source commit IDs.')
    sub.add_parser('example',help='Check tiny saved scoring examples without original sources or a GPU.')
    verify=sub.add_parser('verify-assets');verify.add_argument('--assets',type=Path,required=True)
    prepare=sub.add_parser('prepared',help='Run one prepared 1 s attempt; output is not a latency measurement.')
    prepare.add_argument('--assets',type=Path,required=True);prepare.add_argument('--method',choices=list(configuration()['methods']),required=True)
    prepare.add_argument('--correlation',type=Path,required=True);prepare.add_argument('--variance',type=Path,required=True)
    prepare.add_argument('--output',type=Path,required=True)
    geometry=sub.add_parser('geometry');geometry.add_argument('--assets',type=Path,required=True)
    geometry.add_argument('--scene',type=Path,required=True);geometry.add_argument('--output',type=Path,required=True)
    paced=sub.add_parser('paced',help='Run actual raw frontend and sample-paced selected receiver, with measured availability times.')
    paced.add_argument('--assets',type=Path,required=True);paced.add_argument('--method',choices=list(configuration()['methods']),required=True)
    for key in ['reference','observation','alignment','output']:paced.add_argument('--'+key,type=Path,required=True)
    sample=sub.add_parser('simulation-example',help='Export a recorded scene/seed example and portable generation inputs.')
    sample.add_argument('--output',type=Path,required=True);sample.add_argument('--waveform',type=Path,required=True)
    plan=sub.add_parser('generation-plan');plan.add_argument('--manifest',type=Path,required=True)
    generate=sub.add_parser('generate',help='Explicitly synthesize one recorded episode, or a bounded smoke slice.')
    for key in ['assets','manifest-directory','geometry-root','contract','scratch-root','output']:
        generate.add_argument('--'+key,type=Path,required=True)
    generate.add_argument('--split',required=True);generate.add_argument('--episode-id',required=True)
    generate.add_argument('--duration',type=int,choices=[1,4],default=4)
    generate.add_argument('--smoke-frames',type=int,choices=range(1,6))
    args=p.parse_args()
    if args.command=='methods':print(json.dumps(configuration(),indent=2));return
    if args.command=='example':
        from .examples import scoring_check
        print(json.dumps(scoring_check()));return
    if args.command=='simulation-example':
        from .simulation import write_example
        print(json.dumps(write_example(args.output,args.waveform)));return
    if args.command=='generation-plan':
        from .simulation import generation_plan
        print(json.dumps(generation_plan(args.manifest)));return
    assets=Assets(args.assets)
    if args.command=='verify-assets':
        print(json.dumps(dict(passed=True,files=len(assets.manifest['files']),source_commits=assets.configuration['source_commits'])));return
    if args.command=='generate':
        import numpy as np
        from .simulation import recording_generator,recording_slice
        if args.output.exists():raise FileExistsError('Existing raw output directory is preserved.')
        manifest=json.loads((args.manifest_directory/(args.split+'_manifest.json')).read_text())
        row=next((r for r in manifest['episodes'] if r['episode_id']==args.episode_id),None)
        if row is None:raise ValueError('Episode ID is absent from the supplied manifest.')
        kwargs=dict(manifest_directory=args.manifest_directory,geometry_root=args.geometry_root,contract=args.contract,
                    scratch_root=args.scratch_root,split=args.split)
        values=recording_slice(**kwargs,row=row,frames=args.smoke_frames) if args.smoke_frames else recording_generator(**kwargs,duration_s=args.duration).recording(row)
        args.output.mkdir(parents=True)
        for key,value in zip(['reference','observation','alignment'],values):np.save(args.output/(key+'.npy'),value)
        metadata=dict(episode_id=row['episode_id'],duration_s=args.smoke_frames*.02 if args.smoke_frames else args.duration,
                      sliced_smoke=bool(args.smoke_frames),source_commits=assets.configuration['source_commits'],
                      array_shapes=[list(v.shape) for v in values],array_dtypes=[str(v.dtype) for v in values])
        (args.output/'generation.json').write_text(json.dumps(metadata,indent=2)+'\n')
        print(str(args.output));return
    if args.output.exists():raise FileExistsError('Existing outputs are preserved; select a new output path.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.command=='geometry':
        import numpy as np
        from .geometry import geometry
        cfg=json.loads((assets.root/'physical_configuration.json').read_text())
        arrays,summary=geometry(cfg,json.loads(args.scene.read_text()))
        if args.output.suffix!='.npz':raise ValueError('Geometry output must use .npz.')
        if args.output.with_suffix('.json').exists():raise FileExistsError('Geometry summary already exists.')
        np.savez_compressed(args.output,**arrays)
        args.output.with_suffix('.json').write_text(json.dumps(summary,indent=2)+'\n')
    elif args.command=='paced':
        import numpy as np
        import torch
        from .runtime import Runtime
        from .paced import validate_inputs
        if not torch.cuda.is_available():raise RuntimeError('CUDA is required for selected inference.')
        inputs={k:np.load(getattr(args,k),allow_pickle=False) for k in ['reference','observation','alignment']}
        validate_inputs(**inputs)
        hashes={k:hashlib.sha256(memoryview(a)).hexdigest() for k,a in inputs.items()}
        result=Runtime(args.assets).paced(args.method,**inputs)
        result['input_array_sha256']=hashes
        args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    else:
        import numpy as np
        import torch
        from .runtime import Runtime
        if not torch.cuda.is_available():raise RuntimeError('CUDA is required for selected inference.')
        c=np.load(args.correlation,mmap_mode='r');v=np.load(args.variance,mmap_mode='r')
        if c.shape!=(20000,1792) or v.shape!=(200,1792):raise ValueError('Supply exactly one prepared attempt, not an entire campaign.')
        hashes={name:hashlib.sha256(memoryview(a)).hexdigest() for name,a in [('correlation',c),('variance',v)]}
        result=Runtime(args.assets).prepared(args.method,torch.from_numpy(c.copy()).cuda(),torch.from_numpy(v.copy()).cuda())
        result['input_array_sha256']=hashes
        args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(str(args.output))


if __name__=='__main__':main()
