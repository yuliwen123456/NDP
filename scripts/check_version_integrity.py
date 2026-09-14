#!/usr/bin/env python3
"""Verify immutable version manifests and reject edits to any tracked history.

Standard library only. No training, network calls or mutation. --external also
hashes the preserved server source/checkpoints and therefore takes longer.
"""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for b in iter(lambda:stream.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def verify_manifest(folder):
    errors=[];manifests=list(folder.glob('manifest_*.json'))
    for m in manifests:
        data=json.loads(m.read_text(encoding='utf8'))
        if not data.get('frozen'):continue
        hashes=data['sha256'];allowed=set(hashes)|{m.name}
        for name,want in hashes.items():
            path=folder/name
            try:
                path.resolve().relative_to(folder.resolve())
                if not path.is_file() or digest(path)!=want:errors.append('ERROR frozen file changed/missing: '+str(path))
            except ValueError:errors.append('ERROR manifest path escapes version: '+name)
        actual={str(p.relative_to(folder)).replace('\\','/') for p in folder.rglob('*') if p.is_file()}
        for name in sorted(actual-allowed):errors.append('ERROR new file inside frozen version: '+str(folder/name))
    return errors

def permitted_edit(name,old,new):
    return name=='VERSION_HISTORY.md' and new.startswith(old) and len(new)>len(old)

def git(root,*args):
    return subprocess.run(['git','-C',str(root),*args],capture_output=True)

def check(root,external=False):
    errors=[]
    for folder in sorted((root/'experiments').glob('v*')):
        errors.extend(verify_manifest(folder))
    tracked=git(root,'ls-files','-z')
    if tracked.returncode:errors.append('ERROR not a Git checkout')
    else:
        for raw in tracked.stdout.split(b'\0'):
            if not raw:continue
            name=raw.decode();path=root/name
            old=git(root,'show','HEAD:'+name)
            if old.returncode:continue # a newly staged file
            if not path.is_file():errors.append('ERROR historical file missing: '+name);continue
            new=path.read_bytes()
            if old.stdout!=new and not permitted_edit(name,old.stdout,new):errors.append('ERROR historical file overwritten: '+name)
        # Check index as well; a staged edit cannot hide behind restored working-tree bytes.
        diff=git(root,'diff','--cached','--name-status','--no-renames')
        for line in diff.stdout.decode().splitlines():
            status,name=line.split('\t',1)
            if status=='A':continue
            old=git(root,'show','HEAD:'+name);new=git(root,'show',':'+name)
            if old.returncode==0 and not permitted_edit(name,old.stdout,new.stdout):errors.append('ERROR staged historical edit/deletion: '+name)
        for raw in git(root,'ls-files','--cached','--others','--exclude-standard','-z').stdout.split(b'\0'):
            if not raw:continue
            name=raw.decode();path=root/name
            if path.suffix.lower() in ('.ckpt','.pth','.pt','.safetensors','.bin','.label','.npz'):
                errors.append('ERROR heavyweight/data file in commit scope: '+name)
            if path.is_file() and path.stat().st_size>10*1024*1024:errors.append('ERROR file larger than 10MiB: '+name)
    if external:
        data=json.loads((root/'experiments/v06_baseline/freeze_evidence_v06.json').read_text())
        for name,info in data['external_files'].items():
            p=Path(name)
            if not p.is_file() or digest(p)!=info['sha256']:errors.append('ERROR external historical file changed/missing: '+name)
    status=git(root,'status','--short').stdout.decode(errors='replace')
    print('CURRENT FILE STATUS\n'+(status or '(clean)'))
    for error in errors:print(error)
    print('FAIL' if errors else 'PASS: frozen snapshots and tracked history are intact; no large model/data files in scope.')
    return 1 if errors else 0

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--external',action='store_true')
    a=p.parse_args();sys.exit(check(a.root,a.external))
