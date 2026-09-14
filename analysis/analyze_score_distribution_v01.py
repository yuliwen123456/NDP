"""Read-only, paired diagnostic sample; never tunes or changes official evaluation.

Select evenly spaced frames per sequence BEFORE seeing labels/scores. Use the
official range/ignore/frame eligibility rules. Statistics are sample-only.
"""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score,roc_curve,auc,precision_recall_curve

def stats(x):
    if not len(x):return {'n':0}
    q=np.percentile(x,[5,25,50,75,95])
    return dict(n=int(len(x)),mean=float(x.mean(dtype=np.float64)),std=float(x.std(dtype=np.float64)),
                p05=float(q[0]),p25=float(q[1]),median=float(q[2]),p75=float(q[3]),p95=float(q[4]),
                min=float(x.min()),max=float(x.max()))

def metrics(y,s):
    fpr,tpr,th=roc_curve(y,s);i=np.flatnonzero(tpr>.95)[0]
    pr,re,_=precision_recall_curve(y,s)
    # Report observed precision at first crossing of preset recalls, never deploy thresholds.
    rows={}
    for target in (.1,.25,.5,.75,.9,.95):
        valid=np.flatnonzero(re>=target)
        j=valid[-1]
        rows[str(target)]={'recall':float(re[j]),'precision':float(pr[j])}
    return dict(AP=float(average_precision_score(y,s)*100),FPR95=float(fpr[i]*100),
                AUROC=float(auc(fpr,tpr)*100),diagnostic_threshold=float(th[i]),precision_at_recall=rows)

def run(a):
    if a.output.exists():raise FileExistsError(a.output)
    start=time.time();names=[];packs=[];excluded=[];label_ids=set()
    dirs=sorted(x for x in a.data_dir.glob('1[0-9][0-9]') if x.is_dir())
    for seq in dirs:
        files=sorted((seq/'velodyne').glob('*.bin'))
        indices=np.unique(np.linspace(0,len(files)-1,min(a.frames_per_sequence,len(files)),dtype=int))
        for idx in indices:
            f=files[idx];rel=str(Path(seq.name)/(f.stem+'.txt'));names.append(rel)
            xyz=np.fromfile(f,dtype=np.float32).reshape(-1,4)[:,:3]
            raw=np.fromfile(seq/'labels'/(f.stem+'.label'),dtype=np.uint32)&0xffff
            label_ids.update(int(x) for x in np.unique(raw))
            assert len(xyz)==len(raw)
            dist=np.linalg.norm(xyz,axis=1)
            mask=(raw!=0)&(dist>=2.5)&(dist<=50)
            if np.sum((raw==2)&mask)<5:excluded.append(rel);continue
            s6=np.loadtxt(a.v06/rel,dtype=np.float32);s7=np.loadtxt(a.v07/rel,dtype=np.float32)
            assert len(s6)==len(raw)==len(s7) and np.isfinite(s6).all() and np.isfinite(s7).all()
            # 1-m voxel occupancy is a density proxy, not a kNN estimate or semantic boundary.
            cell=np.floor(xyz/a.density_voxel).astype(np.int32)
            _,inv,cnt=np.unique(cell,axis=0,return_inverse=True,return_counts=True)
            packs.append((s6[mask],s7[mask],raw[mask].astype(np.int16),dist[mask],cnt[inv][mask].astype(np.int32),np.full(mask.sum(),int(seq.name),np.int16)))
        print('SEQUENCE',seq.name,'selected',len(indices),'elapsed',round(time.time()-start,1),flush=True)
    s6,s7,sem,dist,density,seq=np.concatenate([p[0] for p in packs]),np.concatenate([p[1] for p in packs]),np.concatenate([p[2] for p in packs]),np.concatenate([p[3] for p in packs]),np.concatenate([p[4] for p in packs]),np.concatenate([p[5] for p in packs])
    del packs
    y=sem==2
    report={'schema':1,'sampling':'uniform deterministic frames per sequence, before labels/scores; no point subsampling',
            'scope':'DIAGNOSTIC SAMPLE ONLY, not a replacement official result or tuning set',
            'selected_frames':names,'excluded_frames_lt5_ood':excluded,'selected_count':len(names),'eligible_count':len(names)-len(excluded),
            'label_ids_seen':sorted(label_ids),'density_proxy':{'voxel_m':a.density_voxel,'method':'points per axis-aligned voxel'},
            'limitations':['No saved logits, learned NDP weights, confidence or semantic predictions; cannot bin by predicted confidence.',
                          'GT semantic class used only for post-hoc diagnosis, never to fit calibration.',
                          'No semantic-boundary attribution; voxel density is only a proxy.',
                          'Frames within scenes are correlated; no confidence intervals or causal claim.'],
            'filters':{'distance_m':[2.5,50],'ignored_raw_label':0,'ood_raw_label':2,'min_ood_points_per_frame':5},'models':{}}
    # Existing full-evaluation threshold is diagnostic only; it does not change scores or evaluation.
    for name,s,full_threshold in [('v06',s6,a.v06_threshold),('v07',s7,a.v07_threshold)]:
        result={'all_ID':stats(s[~y]),'all_OOD':stats(s[y]),'sample_metrics':metrics(y,s),'groups':{},'full_eval_threshold':full_threshold}
        groups={f'class_{c}':sem==c for c in np.unique(sem)}
        groups.update({f'distance_{lo}_{hi}':(dist>=lo)&(dist<hi if hi<50 else dist<=hi) for lo,hi in [(2.5,10),(10,20),(20,30),(30,50)]})
        groups.update({f'density_{lo}_{hi}':(density>=lo)&(density<hi) for lo,hi in [(1,5),(5,20),(20,100),(100,1000000)]})
        groups.update({f'sequence_{c}':seq==c for c in np.unique(seq)})
        total_fp=int(np.sum((~y)&(s>=full_threshold)))
        for key,m in groups.items():
            idm=m&~y;oodm=m&y;fp=int(np.sum(idm&(s>=full_threshold)))
            result['groups'][key]={'ID':stats(s[idm]),'OOD':stats(s[oodm]),'false_positives':fp,
                                  'fpr_at_full_eval_threshold':float(fp/max(1,idm.sum())),
                                  'share_of_total_fp':float(fp/max(1,total_fp))}
        report['models'][name]=result
    report['score_difference']={'ID_v07_minus_v06':stats((s7-s6)[~y]),'OOD_v07_minus_v06':stats((s7-s6)[y])}
    report['elapsed_seconds']=time.time()-start
    report['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as out:json.dump(report,out,indent=2)
    print('RESULT',a.output, 'eligible_frames',report['eligible_count'], 'points',len(y),flush=True)
    for n,r in report['models'].items():print(n,r['sample_metrics'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data-dir','v06','v07','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--frames-per-sequence',type=int,default=5)
    p.add_argument('--density-voxel',type=float,default=1.)
    p.add_argument('--v06-threshold',type=float,required=True)
    p.add_argument('--v07-threshold',type=float,required=True)
    a=p.parse_args();assert a.frames_per_sequence>0 and a.density_voxel>0
    run(a)
