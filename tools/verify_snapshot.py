"""Static snapshot integrity only; no inference, generation or dependency install."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 data=json.loads((ROOT/'provenance/FILES.json').read_text())
 for row in data['files']:
  p=ROOT/row['path'];assert p.is_file() and not p.is_symlink() and sha(p)==row['sha256'],row['path']
 original=json.loads((ROOT/'artifacts/r2/MANIFEST.json').read_text())
 for name,row in original['artifacts'].items():assert sha(ROOT/'artifacts/r2'/name)==row['sha256'],name
 selected=json.loads((ROOT/'selected_assets/manifest.json').read_text())
 for row in selected['files']:assert sha(ROOT/'selected_assets'/row['path'])==row['sha256'],row['path']
 config=json.loads((ROOT/'src/leo_ofdm_repro/_config/methods.json').read_text())
 assert {r['path']:r['sha256'] for r in selected['files']}==config['asset_sha256']
 print(json.dumps(dict(passed=True,files=len(data['files']),bytes=sum(r['bytes'] for r in data['files']),experiments_run=False)))
if __name__=='__main__':main()
