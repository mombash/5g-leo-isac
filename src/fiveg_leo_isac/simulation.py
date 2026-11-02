"""Manifest-driven propagation and noise synthesis.

Recording synthesis may create large arrays. It is an explicit API, not invoked
by package installation, examples or receiver checks. Preserve recorded seeds.
"""
import ast
import json
from pathlib import Path
from . import simulation_data as data
from .duration_adapters import ShortData


class DataLayout:
    def __init__(self,manifest_directory,geometry_root,contract):
        self.manifests=Path(manifest_directory);self.geometry=Path(geometry_root);self.contract=Path(contract)
    def __truediv__(self,key):
        if key=='geometry':return self.geometry
        if key=='contract.json':return self.contract
        return self.manifests/key


def recording_generator(*,manifest_directory,geometry_root,contract,scratch_root,split,duration_s=4):
    data.OUT=DataLayout(manifest_directory,geometry_root,contract)
    data.CACHE=Path(scratch_root).resolve()
    if duration_s==4:return portable_waveforms(data.EpisodeData(split),manifest_directory)
    if duration_s!=1:raise ValueError('Only the evaluated 4 s and 1 s recording forms are supported.')
    # Resize the recording arrays to the one-second cohort duration.
    import inspect
    tree=ast.parse(inspect.getsource(data.EpisodeData))
    node=ShortData().visit(tree.body[0]);namespace=dict(data.__dict__)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),data.__file__,'exec'),namespace)
    return portable_waveforms(namespace['EpisodeData'](split),manifest_directory)


def portable_waveforms(generator,manifest_directory):
    for waveform in generator.manifest['waveforms']:
        path=Path(waveform['cache_path'])
        if not path.is_absolute():waveform['cache_path']=str((Path(manifest_directory)/path).resolve())
    return generator


def generation_plan(manifest):
    m=json.loads(Path(manifest).read_text());duration=m.get('duration_s',4)
    if duration not in [1,4]:raise ValueError('Unexpected recorded duration.')
    return dict(duration_s=duration,recordings=len(m['episodes']),waveforms=len(m['waveforms']),
                raw_complex64_bytes_per_record=round((2*duration+.01)*30.72e6)*8,
                scientific_inputs='Recorded scenes, waveform/cell/payload/control/noise seeds, geometry, reference choice and physical contract.',
                generated=False)


def recording_slice(*,manifest_directory,geometry_root,contract,scratch_root,split,row,frames=1):
    """Synthesize a short diagnostic slice of 1–5 twenty-ms frames.

    Use the supplied waveform and recorded seeds. A slice is too short for
    a one-second receiver attempt.
    """
    if type(frames) is not int or not 1<=frames<=5:raise ValueError('Slice requires 1–5 twenty-ms frames.')
    data.OUT=DataLayout(manifest_directory,geometry_root,contract)
    data.CACHE=Path(scratch_root).resolve()
    import inspect
    class Slice(ast.NodeTransformer):
        def visit_Constant(self,node):
            if type(node.value) in (int,float) and node.value in (4,4.01,200):
                return ast.copy_location(ast.Constant({4:frames*.02,4.01:frames*.02+.01,200:frames}[node.value]),node)
            return node
    node=Slice().visit(ast.parse(inspect.getsource(data.EpisodeData)).body[0]);namespace=dict(data.__dict__)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),data.__file__+'#bounded-slice','exec'),namespace)
    return portable_waveforms(namespace['EpisodeData'](split),manifest_directory).recording(row)


def write_example(output,waveform_path):
    """Export one recorded scene/seed example and its portable input contract."""
    from importlib.resources import files
    output=Path(output)
    if output.exists():raise FileExistsError('Existing example directory is preserved.')
    example=json.loads(files('fiveg_leo_isac').joinpath('_config/simulation_example.json').read_text())
    manifest=example['manifest'];waveform=dict(manifest['waveforms'][0]);waveform['cache_path']=str(Path(waveform_path).resolve())
    manifest['waveforms']=[waveform];output.mkdir(parents=True)
    (output/(manifest['split']+'_manifest.json')).write_text(json.dumps(manifest,indent=2)+'\n')
    (output/'scene.json').write_text(json.dumps(manifest['scenes'][0],indent=2)+'\n')
    (output/'waveform.json').write_text(json.dumps({k:waveform[k] for k in ['cell_id','payload_seed','control_seed','waveform_id']},indent=2)+'\n')
    (output/'contract.json').write_text(json.dumps(example['contract'],indent=2)+'\n')
    metadata=dict(split=manifest['split'],episode_id=manifest['episodes'][0]['episode_id'],scene_id=manifest['scenes'][0]['scene_id'],
                duration_s=manifest.get('duration_s',4),records=len(manifest['episodes']),purpose='One recorded simulation scene and its waveform/noise seeds.',generation_performed=False)
    (output/'inputs.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return metadata


def prepare_geometry(assets,inputs,output):
    """Write geometry for the first scene in an exported input directory."""
    from .geometry import geometry
    metadata=json.loads((Path(inputs)/'inputs.json').read_text())
    scene=json.loads((Path(inputs)/'scene.json').read_text())
    destination=Path(output)/metadata['split']/(metadata['scene_id']+'.npz')
    if destination.exists() or destination.with_suffix('.json').exists():
        raise FileExistsError('Existing geometry is preserved.')
    cfg=json.loads((assets.root/'physical_configuration.json').read_text())
    arrays,summary=geometry(cfg,scene)
    import numpy as np
    destination.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(destination,**arrays)
    destination.with_suffix('.json').write_text(json.dumps(summary,indent=2)+'\n')
    return dict(geometry=str(destination),scene_id=metadata['scene_id'])
