"""Verify the reviewed tree and archived/current delivery hashes; no inference."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 data=json.loads((ROOT/'provenance/FILES.json').read_text())
 for row in data['files']:
  p=ROOT/row['path'];assert p.is_file() and not p.is_symlink() and sha(p)==row['sha256'],row['path']
 for version in ['r2','v18']:
  original=json.loads((ROOT/'artifacts'/version/'MANIFEST.json').read_text())
  for name,row in original['artifacts'].items():assert sha(ROOT/'artifacts'/version/name)==row['sha256'],(version,name)
 selected=json.loads((ROOT/'selected_assets/manifest.json').read_text())
 for row in selected['files']:assert sha(ROOT/'selected_assets'/row['path'])==row['sha256'],row['path']
 config=json.loads((ROOT/'src/leo_ofdm_repro/_config/methods.json').read_text());assert config['submission']['id']=='V18'
 assert {r['path']:r['sha256'] for r in selected['files']}==config['asset_sha256']
 print(json.dumps(dict(passed=True,submission='V18',files=len(data['files']),bytes=sum(r['bytes'] for r in data['files']),experiments_run=False)))
if __name__=='__main__':main()
