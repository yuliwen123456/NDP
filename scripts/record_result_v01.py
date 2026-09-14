"""Record a NEW completed result and append history; never change proposal files."""
import argparse,datetime,json,math,subprocess
from pathlib import Path
from check_version_integrity import check

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--version',required=True);p.add_argument('--metrics',type=Path,required=True)
    p.add_argument('--conclusion',choices=['KEEP','DROP','NEEDS_MORE_TESTING'],required=True)
    a=p.parse_args(); root=Path(__file__).resolve().parents[1]
    if not a.version.startswith('v') or not a.version.replace('_','').isalnum():p.error('Invalid version')
    if check(root):raise SystemExit('Integrity check failed')
    raw=json.loads(a.metrics.read_text());baseline=json.loads((root/'experiments/v06_baseline/metrics.json').read_text())
    values={k:float(raw[k]) for k in ('AP','FPR95','AUROC')}
    if not all(math.isfinite(v) and 0<=v<=100 for v in values.values()):p.error('Metrics must be finite percentages')
    # Require explicit evaluation provenance; no fabricated default split or epoch.
    for key in ('checkpoint','split','evaluation_command'):
        if not raw.get(key):p.error('Missing provenance: '+key)
    sha=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    result={**raw,**values,'version':a.version,'code_commit':sha,'conclusion':a.conclusion,
            'delta_vs_v06':{k:values[k]-baseline[k] for k in values},'recorded_utc':stamp}
    folder=root/'results';folder.mkdir(exist_ok=True)
    dest=folder/(a.version+'_'+stamp+'.json')
    with dest.open('x',encoding='utf8') as f:json.dump(result,f,indent=2)
    with (root/'VERSION_HISTORY.md').open('a',encoding='utf8',newline='\n') as f:
        f.write('\n## '+a.version+' result '+stamp+'\n\n'+json.dumps(result,ensure_ascii=False)+'\n')
    print(dest)

if __name__=='__main__':main()
