"""Verify the model, policy and physical-configuration files before use."""
import hashlib
import json
from pathlib import Path
from importlib.resources import files


def configuration():
    return json.loads(files('leo_ofdm_repro').joinpath('_config/methods.json').read_text())


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


class Assets:
    def __init__(self,root):
        self.root=Path(root).expanduser().resolve(strict=True)
        self.configuration=configuration()
        self.manifest=json.loads((self.root/'manifest.json').read_text())
        if {r['path']:r['sha256'] for r in self.manifest['files']}!=self.configuration['asset_sha256']:
            raise RuntimeError('Assets do not match this package configuration.')
        for r in self.manifest['files']:
            path=self.root/r['path']
            if self.root not in path.resolve().parents:raise ValueError('Asset escapes its root.')
            if digest(path)!=r['sha256']:raise RuntimeError('Asset hash mismatch: '+r['path'])
        if self.manifest['source_commits']!=self.configuration['source_commits']:
            raise RuntimeError('Asset/configuration source versions differ.')
    @property
    def models(self):return self.root/'models'
    @property
    def policies(self):return self.root/'policies'
