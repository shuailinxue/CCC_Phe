import json
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata
from phenoniche.v1002.lr_atlas import load_lr_atlas
from phenoniche.v1002.simulation_cells import build_simulation_spec, simulate_spatial
from phenoniche.v1002.simulation_model import fit_multiview, _cosine_rows, _correlation


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else float(x)))


def prepare(rep, purity=.5, size=100, noise=.05):
    spec = build_simulation_spec(load_lr_atlas('data/commuspace_human_lr_atlas.tsv'))
    s = simulate_spatial(spec, purity, size, noise, 31000 + rep + int(purity*1000) + size*17 + int(noise*10000))
    f = np.flatnonzero(s.final_mask.ravel())
    c = np.sqrt(np.mean(s.cs ** 2))
    si = np.sqrt(np.mean(s.communication[:, f] ** 2)) / c
    so = np.sqrt(np.mean(s.os ** 2)) / c
    blocks = {'HC': s.cs, 'HI': s.communication[:, f] / si, 'HO': s.os / so}
    ws = np.zeros((1800, 6))
    for i in range(1800):
        cells, weights, _ = s.neighborhoods.members(i)
        ws[i] = np.bincount(s.labels[cells], weights=weights, minlength=6) / weights.sum()
    truth = {'HC': np.vstack([np.full(8, .125), s.hc_truth]),
             'HI': np.vstack([blocks['HI'][s.labels == 0].mean(0), spec.hi_truth[:, f] / si]),
             'HO': np.stack([blocks['HO'][s.labels == k].mean(0) for k in range(6)])}
    return s, blocks, truth, ws


def evaluate(w, h, truth, s, ws):
    similarity = sum(_cosine_rows(h[k], truth[k]) for k in h if k in ('HC','HI')) / sum(k in ('HC','HI') for k in h)
    r, c = linear_sum_assignment(-similarity)
    order = np.empty(6, int); order[c] = r
    aligned = w[:, order]
    normalized = aligned / np.maximum(aligned.sum(1, keepdims=True), 1e-12)
    pred = normalized.argmax(1)
    rows = []
    for k in range(6):
        target = s.labels == k
        ranks = rankdata(aligned[:, k])
        n = target.sum(); m = (~target).sum()
        rows.append({'sensitivity': float(np.mean(pred[target] == k)), 'specificity': float(np.mean(pred[~target] != k)),
                     'auc': float((ranks[target].sum()-n*(n+1)/2)/(n*m)), 'ws': _correlation(aligned[:, k], ws[:, k])})
    sim = _cosine_rows(h['HI'], truth['HI'][1:3])
    return {'per_factor': rows, 'collision': bool(sim[:,0].argmax()==sim[:,1].argmax()), 'overall_ws': float(np.mean([x['ws'] for x in rows])),
            'overall_sensitivity': float(np.mean([x['sensitivity'] for x in rows])), 'overall_auc': float(np.mean([x['auc'] for x in rows])),
            'order':order.tolist(), 'predicted':pred.tolist(), 'activity':normalized.tolist()}


def passed(reports):
    return all(not x['collision'] and x['per_factor'][0]['sensitivity'] >= .8 and all(x['per_factor'][k]['sensitivity'] >= .8 and x['per_factor'][k]['auc'] >= .85 for k in (1,2)) for x in reports)


def audit(w, h, blocks):
    result = {}
    for name, x in blocks.items():
        d = h[name]
        residual = w @ d - x
        grad = 2 * (residual @ d.T) / x.numel()
        result[name] = {'loss':float(residual.square().mean()), 'gradient_norm':float(grad.norm())}
    gi = result['HI']['gradient_norm']
    result['ratios_C_I_O'] = [result['HC']['gradient_norm']/max(gi,1e-30),1.,result['HO']['gradient_norm']/max(gi,1e-30)]
    return result


def warm(ccc, blocks, truth, s, ws):
    device='cuda'
    x = {k:torch.tensor(v,device=device) for k,v in blocks.items()}
    w=ccc.W.to(device).clone(); hi=ccc.dictionaries['HI'].to(device).clone()
    scale=w.mean().clamp_min(1e-12)
    w/=scale; hi*=scale
    h={'HI':hi, 'HC':torch.full((6,8),.1,device=device),'HO':torch.full((6,24),.1,device=device)}
    ref=w.clone()
    stage1=evaluate(w.cpu().numpy(),{'HI':hi.cpu().numpy()},truth,s,ws)
    twin=stage1['order'][1:3]
    history=[]
    def record(stage, epoch):
        a=audit(w,h,x)
        a.update(stage=stage,epoch=epoch,cosine_to_CCC=[float(torch.nn.functional.cosine_similarity(w[:,k],ref[:,k],dim=0)) for k in twin],
                 HI_twin_cosine=float(torch.nn.functional.cosine_similarity(h['HI'][twin[0]],h['HI'][twin[1]],dim=0)))
        history.append(a)
    record(2,0)
    for epoch in range(1,501):
        gram=w.T@w
        for name in ('HC','HO'):
            h[name]*=(w.T@x[name])/(gram@h[name]+1e-8)
        record(2,epoch)
    staged=(w.clone(),{k:v.clone() for k,v in h.items()})
    stage2=evaluate(w.cpu().numpy(),{k:v.cpu().numpy() for k,v in h.items()},truth,s,ws)
    onset=None
    spatial_onset=None
    for epoch in range(1,501):
        gram=w.T@w
        for name in h:
            ratio=(w.T@x[name])/(gram@h[name]+1e-8)
            h[name]*=ratio.pow(.1)
        numerator=sum(x[k]@h[k].T/x[k].shape[1] for k in h)
        denominator=w@sum(h[k]@h[k].T/x[k].shape[1] for k in h)
        w*=(numerator/(denominator+1e-8)).pow(.1)
        record(3,epoch)
        pred = w[:, stage2['order']].argmax(1).cpu().numpy()
        sens = [float(np.mean(pred[s.labels==k] == k)) for k in (0,1,2)]
        history[-1]['fixed_matching_sensitivity_bg_N1_N2'] = sens
        if spatial_onset is None and min(sens) < .8:
            spatial_onset=epoch
        sim=torch.nn.functional.normalize(h['HI'],dim=1)@torch.nn.functional.normalize(torch.tensor(truth['HI'][1:3],device=device),dim=1).T
        if onset is None and int(sim[:,0].argmax())==int(sim[:,1].argmax()): onset=epoch
    final=evaluate(w.cpu().numpy(),{k:v.cpu().numpy() for k,v in h.items()},truth,s,ws)
    return stage1, stage2, final, history, {'HI_collision_epoch':onset,'spatial_gate_loss_epoch':spatial_onset}


def run():
    out=Path('outputs/v1002_repair');out.mkdir(exist_ok=True,parents=True)
    datasets=[]; reports={'Full-Random-K6':[],'CCC-only-K6':[]}; ccc_models=[]
    for rep in range(3):
        s,b,t,ws=prepare(rep); datasets.append((s,b,t,ws))
        for name,blocks in [('Full-Random-K6',b),('CCC-only-K6',{'HI':b['HI']})]:
            traces=[]
            xb={k:torch.tensor(v,device='cuda') for k,v in b.items()}
            callback=(lambda epoch,w,h: traces.append({'epoch':epoch,**audit(w,h,xb)})) if name=='Full-Random-K6' else None
            fit=fit_multiview(blocks,niches=6,seed=31+rep,audit_callback=callback)
            if traces:save(out/f'random_audit_seed{31+rep}.json',{'checkpoints':traces,'optimizer':'multiplicative updates','iterations':500,'early_stopping':None})
            report=evaluate(fit.W.numpy(),{k:v.numpy() for k,v in fit.dictionaries.items()},t,s,ws)
            reports[name].append(report)
            if name=='CCC-only-K6':ccc_models.append(fit)
        print('K6 seed',31+rep,{k:[r['sensitivity'] for r in v[-1]['per_factor'][:3]] for k,v in reports.items()},flush=True)
    save(out/'k6.json',reports)
    cause='A' if passed(reports['Full-Random-K6']) else 'E'
    if cause!='A':
        reports['Full-CCC-WarmStart-K6']=[]; reports['Staged-K6']=[]
        for rep,(s,b,t,ws) in enumerate(datasets):
            a,st,joint,hist,onset=warm(ccc_models[rep],b,t,s,ws)
            reports['Full-CCC-WarmStart-K6'].append(joint);reports['Staged-K6'].append(st)
            save(out/f'optimization_seed{31+rep}.json',{'stage1':a,'stage2':st,'joint':joint,'history':hist,'collapse_onset':onset,
                 'view_statistics':{k:{'mean':float(v.mean()),'std':float(v.std()),'RMS':float(np.sqrt(np.mean(v*v))),'sparsity':float(np.mean(v==0))} for k,v in b.items()}})
            print('warm seed',31+rep,'stage2', [r['sensitivity'] for r in st['per_factor'][:3]],'joint',[r['sensitivity'] for r in joint['per_factor'][:3]],flush=True)
        if passed(reports['Full-CCC-WarmStart-K6']):cause='B'
        elif passed(reports['Staged-K6']) and not passed(reports['Full-CCC-WarmStart-K6']):cause='C'
    if cause=='E':
        reports['Full-Random-RMS-K6']=[];reports['Full-WarmStart-RMS-K6']=[]
        for rep,(s,b,t,ws) in enumerate(datasets):
            rms={k:float(np.sqrt(np.mean(v*v))) for k,v in b.items()}
            scaled={k:v/rms[k] for k,v in b.items()}
            scaled_truth={k:v/rms[k] for k,v in t.items()}
            fit=fit_multiview(scaled,niches=6,seed=31+rep)
            reports['Full-Random-RMS-K6'].append(evaluate(fit.W.numpy(),{k:v.numpy() for k,v in fit.dictionaries.items()},scaled_truth,s,ws))
            ccc=fit_multiview({'HI':scaled['HI']},niches=6,seed=31+rep)
            a,st,joint,hist,onset=warm(ccc,scaled,scaled_truth,s,ws)
            reports['Full-WarmStart-RMS-K6'].append(joint)
            save(out/f'rms_audit_seed{31+rep}.json',{'scales':rms,'joint':joint,'history':hist,'collapse_onset':onset})
            print('RMS seed',31+rep,[r['sensitivity'] for r in joint['per_factor'][:3]],flush=True)
        if passed(reports['Full-WarmStart-RMS-K6']):cause='D'
    final={'D':'Full-WarmStart-RMS-K6','A':'Full-Random-K6','B':'Full-CCC-WarmStart-K6','C':'Staged-K6'}.get(cause,'Full-CCC-WarmStart-K6')
    save(out/'summary.json',{'cause':cause,'final_method':final,'ST_gate_passed':passed(reports[final]),'reports':reports,'bulk':'not_run_ST_gate_failed' if not passed(reports[final]) else 'pending'})
    np.savez(out/'primary_maps.npz',coordinates=datasets[0][0].coordinates,labels=datasets[0][0].labels,hc=datasets[0][2]['HC'],hi=datasets[0][2]['HI'])

if __name__=='__main__':run()
