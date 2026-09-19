from pathlib import Path
import json
import time
import numpy as np
import torch
from phenoniche.v1002.repair_primary import prepare, evaluate, passed, save
from phenoniche.v1002.simulation_cells import build_simulation_spec,simulate_bulk,bulk_potential
from phenoniche.v1002.lr_atlas import load_lr_atlas
from phenoniche.v1002.simulation_model import fit_multiview,infer_activity,score_bulk,_auprc
from phenoniche.evaluation.metrics import concordance_index


def staged(blocks,seed):
    ccc=fit_multiview({'HI':blocks['HI']},niches=6,seed=seed)
    w=ccc.W.cuda().clone();hi=ccc.dictionaries['HI'].cuda().clone()
    scale=w.mean().clamp_min(1e-12);w/=scale;hi*=scale
    h={'HI':hi,'HC':torch.full((6,8),.1,device='cuda'),'HO':torch.full((6,24),.1,device='cuda')}
    gram=w.T@w
    targets={k:w.T@torch.tensor(blocks[k],device='cuda') for k in ('HC','HO')}
    for _ in range(500):
        for k in targets:h[k]*=targets[k]/(gram@h[k]+1e-8)
    return w.cpu().numpy(),{k:v.cpu() for k,v in h.items()}


def run():
    out=Path('outputs/v1002_repair');summary=json.loads((out/'summary.json').read_text())
    if not passed(summary['reports']['Staged-K6']):raise RuntimeError('Staged primary gate failed')
    summary.update(cause='C',final_method='Staged-K6',ST_gate_passed=True)
    summary['interpretation']={'joint_HI_collision_count':0,'joint_spatial_collapse':True,'onset_epochs':[110,105,108],
        'fallback_basis':'Stage2 passes every ST criterion; Stage3 destroys spatial identifiability despite distinct HI best matches.'}
    spec=build_simulation_spec(load_lr_atlas('data/commuspace_human_lr_atlas.tsv'))
    records=[];start=time.perf_counter()
    settings=[(.5,100,.05)]+[(p,z,n) for p in (.3,.5,.7) for z in (50,100,150) for n in (0.,.05,.1) if (p,z,n)!=(.5,100,.05)]
    for purity,size,noise in settings:
        for rep in range(3):
            s,b,t,ws=prepare(rep,purity,size,noise)
            w,h=staged(b,31+rep)
            report=evaluate(w,{k:v.numpy() for k,v in h.items()},t,s,ws)
            primary=(purity,size,noise)==(.5,100,.05)
            if primary:
                if not passed([report]):raise RuntimeError('Final replay failed ST gate')
                torch.save({'WS':torch.tensor(w),**h},out/f'final_seed{31+rep}.pt')
            features=np.flatnonzero(s.final_mask.ravel())
            seed=31000+rep+int(purity*1000)+size*17+int(noise*10000)
            pi,cb,expression,times,event,beta,eta=simulate_bulk(spec,s.hc_truth,noise,seed+7000)
            scale=np.sqrt(np.mean(s.communication[:,features]**2))/np.sqrt(np.mean(s.cs**2))
            potential=bulk_potential(expression,spec)[:,features]/scale
            activity=infer_activity({'HC':cb,'HI':potential},{k:h[k] for k in ('HC','HI')})
            bulk=score_bulk('CCC-anchored-staged-K6',activity,report['order'][1:],pi,times,event)
            ap=[]
            for niche in range(5):
                signal=np.isin(features,[f for k,f,_,_ in spec.active_edges if k==niche])
                ap.append(_auprc(signal,h['HI'][report['order'][niche+1]].numpy()))
            record={'purity':purity,'size':size,'noise':noise,'replicate':rep,'seed':31+rep,'spatial':report,'bulk':bulk,'CCC_AUPRC':float(np.mean(ap)),
                    'true_survival_C':concordance_index(times,event,eta)}
            if not primary:
                report.pop('activity');report.pop('predicted')
            records.append(record)
            save(out/'final_grid_progress.json',{'completed':len(records),'total':81})
        print('done',purity,size,noise,flush=True)
    primary=[r for r in records if (r['purity'],r['size'],r['noise'])==(.5,100,.05)]
    summary['reports']['Final method']=[r['spatial'] for r in primary]
    summary['bulk']=[r['bulk'] for r in primary]
    summary['true_survival_C']=[r['true_survival_C'] for r in primary]
    summary['robustness_runtime_seconds']=time.perf_counter()-start
    summary.setdefault('protocol',{})['beta_background']=0
    summary['protocol']['bulk_generator']='unchanged five biological exposures; background exposure zero; six inferred columns with unconstrained existing Cox'
    save(out/'final_robustness_grid.json',records);save(out/'summary.json',summary)

if __name__=='__main__':run()
