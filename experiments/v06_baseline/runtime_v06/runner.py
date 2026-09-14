"""Prepare and execute sixth BN running-stat comparison."""
import sys, os, json, subprocess, datetime, hashlib
from pathlib import Path
ROOT=Path('/root/autodl-tmp/NDP')
HERE=Path(__file__).resolve().parent
PY=ROOT/'envs/ndp/bin/python'
PREVIOUS=ROOT/'outputs/compare-loss-corrected-b2-20260913-194933'
sys.path.insert(0,str(ROOT/'outputs/next-round-preparation-20260912'))
from next_round import execute

def source_hashes():
    return {str(p.relative_to(ROOT/'code/ndp')):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROOT/'code/ndp').rglob('*') if p.is_file() and p.suffix in ('.py','.yaml')}

def main():
    if sys.argv[1]=='--prepare':
        old=json.loads((PREVIOUS/'plan.json').read_text())
        out=ROOT/'outputs'/('compare-bn-ema-b2-'+datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
        plan=json.loads(json.dumps(old).replace(old['output'],str(out)))
        plan.update(arm='bn-ema-b2',comparison_baseline=str(PREVIOUS),
                    objective='Relative to fifth, change ONLY backbone BN running-stat retention per accumulation group; preserve fifth loss correction. Not physical-batch8 normalization.',
                    correction='Criterion sample-sum terms times actual accumulation-group length; total loss times 4/group length. Includes auxiliary losses and epoch tail.',
                    remaining_differences=['micro-batch BatchNorm','mean of micro-batch point losses vs point-weighted global mean','batch-dependent sampling'],
                    entry=str(HERE/'train.py'))
        train_cmd=[str(PY),'-B',str(HERE/'train.py'),'formal',
                   'general.save_dir='+str(out/'training'),'hydra.run.dir='+str(out/'hydra')]
        plan['bn_policy']='62 backbone BNs: preserve original momenta individually; use 1-(1-original_m)**(1/actual_group_size). Original .02:16 modules; .1:46 modules. Full group4; tail group1.'
        plan['stages'][0]['command']=train_cmd
        plan['training_overrides']=['Use saved fourth baseline.yaml; physical=2, accumulation=4, LR=2e-4, seed=2025, epochs=10, OneCycle=1130; runtime loss compensation v1 plus BN group-retention policy v1']
        with (HERE/'plan.json').open('x') as f: json.dump(plan,f,indent=2)
        with (HERE/'source-before.json').open('x') as f: json.dump(source_hashes(),f,indent=2)
        print(json.dumps(plan,indent=2))
        return 0
    assert sys.argv[1]=='--execute'
    plan=json.loads((HERE/'plan.json').read_text())
    previous_status=json.loads((PREVIOUS/'status.json').read_text())
    assert previous_status['status']=='completed' and all(s.get('exit_code')==0 for s in previous_status['stages'])
    with (HERE/'launch-claim.json').open('x') as f:
        json.dump(dict(pid=os.getpid(),started=datetime.datetime.now().isoformat(),output=plan['output']),f)
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='0',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',HYDRA_FULL_ERROR='1')
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    row=[r for r in rows.splitlines() if r.split(',')[0].strip()=='0'][0].split(',')
    assert int(row[1])<512 and int(row[2])==0, 'GPU0 is busy; refusing to start'
    assert source_hashes()==json.loads((HERE/'source-before.json').read_text())
    subprocess.run([str(PY),'-B',str(HERE/'loss_correction.py')],check=True,env=env)
    subprocess.run([str(PY),'-B',str(HERE/'bn_policy.py')],check=True,env=env)
    command=['/usr/local/bin/train-monitor','run','--',str(PY),'-B',str(HERE/'train.py'),'preflight',
             'general.save_dir='+str(HERE/'preflight/training'),
             'hydra.run.dir='+str(HERE/'preflight/hydra'),'+trainer.max_steps=10']
    print('PREFLIGHT_COMMAND '+json.dumps(command),flush=True)
    result=subprocess.run(command,env=env,cwd=ROOT)
    if result.returncode:
        print('Preflight failed. Formal training was not started.',flush=True)
        return result.returncode
    evidence=json.loads((HERE/'preflight/preflight-result.json').read_text())
    assert evidence['success'] and evidence['optimizer_steps']==10 and evidence['training_batches']==40
    assert evidence['gradient_checks']>0
    assert evidence['peak_reserved_gib']<22, 'Insufficient memory margin'
    assert source_hashes()==json.loads((HERE/'source-before.json').read_text())
    print('PREFLIGHT_PASSED; starting fresh formal training from original STU initialization.',flush=True)
    code=execute(plan)
    after=source_hashes()
    (HERE/'source-after.json').write_text(json.dumps(after,indent=2))
    assert after==json.loads((HERE/'source-before.json').read_text())
    return code

if __name__=='__main__':
    sys.exit(main())
