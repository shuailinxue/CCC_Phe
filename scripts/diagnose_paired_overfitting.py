"""Read-only sampled probes of the paired-tissue model; never replace results.

Continue the same best checkpoint with fresh, matched optimizers. These are
diagnostic validation probes, not new unbiased test scores or hyperparameter
selection for the delivered experiment. No test slices are loaded on GPU.
"""
from pathlib import Path
from dataclasses import replace
from functools import partial
import copy,json,time
import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from phenoniche.gnn.data import GraphData,Targets
from phenoniche.gnn.config import GNNConfig
from phenoniche.gnn.pipeline import load_checkpoint
from phenoniche.gnn.training import epoch,make_optimizer,initialize
from phenoniche.gnn.model import model_from_config
from phenoniche.gnn.losses import ccc_loss
from phenoniche.data import feature_statistics
from phenoniche.training import set_seed
from phenoniche.simulation.response_simulation import checksum

ROOT=Path('/home/nas3/biod/xueshuailin/CCC_Phe/workspace/outputs/healthy_cancer_communication')


def main():
    out=ROOT/'diagnostics/overfitting';out.mkdir(parents=True,exist_ok=True)
    sources={f:checksum(ROOT/f) for f in ('gnn/checkpoint.pt','gnn/Z.npy','gnn/H.npy','metrics.csv','assignments.csv.gz')}
    cfg=GNNConfig.from_json(ROOT/'gnn/config.json')
    assert cfg.device=='cuda:2';torch.set_num_threads(6);set_seed(cfg.seed)
    a=ad.read_h5ad(ROOT/'tissues.h5ad');raw=np.asarray(a.X,np.float32)
    prep=np.load(ROOT/'input/preprocessing.npz');folds=np.load(ROOT/'input/folds.npy')
    members=np.load(ROOT/'input/members.npy');weights=np.load(ROOT/'input/weights.npy')
    rows=[np.flatnonzero(folds==i) for i in range(3)];categories=tuple(pd.read_csv(ROOT/'niche_composition.csv',index_col=0).columns)
    types=pd.Categorical(a.obs.cell_type,categories=categories).codes.astype(np.int64)
    c=np.load(ROOT/'input/ccc.npy',mmap_mode='r');mean,_,_,nz=feature_statistics(c,rows[0])
    data=GraphData(sp.csr_matrix(raw),raw.sum(1),tuple(a.var_names),prep['encoder_indices'],prep['gene_mean'],prep['gene_scale'],
                   types,categories,np.asarray(a.obsm['spatial'],np.float32),members,weights,Targets(c,np.arange(c.shape[1])),
                   prep['feature_scale'],prep['feature_mask'],pd.read_csv(ROOT/'input/features.csv'),
                   a.obs[['cell_id','cell_type']].reset_index(drop=True),rows,folds,a.obs.pair_id.to_numpy(),
                   np.load(ROOT/'input/composition.npy'),json.loads((ROOT/'input/audit.json').read_text()),nz,mean)
    model,payload=load_checkpoint(data,cfg);device=torch.device('cuda:2');model.to(device)
    state=copy.deepcopy(model.state_dict());initial_h=model.dictionary.detach().cpu().numpy().copy()
    rng=np.random.default_rng(8172);sample=a.obs.sample_id.to_numpy()
    fit=[];same=[];val=[]
    for s in np.unique(sample):
        ids=np.flatnonzero(sample==s)
        if folds[ids[0]]==0:
            chosen=rng.choice(ids,500,replace=False);fit.extend(chosen)
            same.extend(rng.choice(np.setdiff1d(ids,chosen),128,replace=False))
        elif folds[ids[0]]==1:val.extend(rng.choice(ids,500,replace=False))
    fit=np.sort(fit);same=np.sort(same);val=np.sort(val);monitor=rng.choice(fit,512,replace=False)
    data.prepare_subset_device(np.unique(np.r_[fit,same,val]),device)
    assert (folds[data._resident_cell_ids]!=2).all()
    expression=data._resident['expression'];original=expression.clone()
    links=pd.read_csv(ROOT/'lr.csv');lr_genes=set(links.ligand)|set(links.receptor)
    is_lr=np.array([data.genes[i] in lr_genes for i in data.encoder_indices]+[False])
    non_lr=np.array([data.genes[i] not in lr_genes for i in data.encoder_indices]+[False])
    lr_idx=torch.tensor(np.flatnonzero(is_lr),device=device);bg_idx=torch.tensor(np.flatnonzero(non_lr),device=device)
    resident_types=types[data._resident_cell_ids];resident_samples=sample[data._resident_cell_ids]
    resident_fold=folds[data._resident_cell_ids]
    objective=partial(ccc_loss,kind='huber',delta=3.)

    def evaluate(m,r):return epoch(m,data,r,cfg,ccc_objective=objective)['ccc']

    def interventions(m,name):
        records=[]
        for kind in ('original','background_type_mean','background_shuffle_within_sample','LR_shuffle_within_sample'):
            expression.copy_(original)
            generator=np.random.default_rng(918)
            if kind=='background_type_mean':
                for t in range(len(categories)):
                    donors=np.flatnonzero((resident_types==t)&(resident_fold==0))
                    ids=torch.tensor(np.flatnonzero(resident_types==t),device=device)
                    means=original[torch.tensor(donors,device=device)][:,bg_idx].mean(0)
                    expression[ids[:,None],bg_idx[None,:]]=means
            elif 'shuffle' in kind:
                columns=lr_idx if kind.startswith('LR') else bg_idx
                for s in np.unique(resident_samples):
                    for t in range(len(categories)):
                        ids=np.flatnonzero((resident_types==t)&(resident_samples==s));donors=generator.permutation(ids)
                        ii=torch.tensor(ids,device=device);dd=torch.tensor(donors,device=device)
                        expression[ii[:,None],columns[None,:]]=original[dd[:,None],columns[None,:]]
            records.append({'model':name,'intervention':kind,'train_CCC':evaluate(m,monitor),
                            'same_slice_control_CCC':evaluate(m,same),'new_slice_validation_CCC':evaluate(m,val)})
        expression.copy_(original)
        return records

    facts={'total_parameters':sum(p.numel() for p in model.parameters()),'first_expression_layer_parameters':sum(p.numel() for p in model.cell_encoder[0].parameters()),
           'genes':len(data.genes),'LR_genes':int(is_lr.sum()),'original_training_slices':4,'independent_training_geometries':2,
           'sampled_training_centers':len(fit),'training_monitor_centers':len(monitor),'same_slice_control_centers':len(same),
           'new_slice_validation_centers':len(val),'test_used':False,'checkpoint_epoch':payload['best_epoch'],
           'same_slice_control_is_overlapping_not_independent':True,'optimizer':'Fresh AdamW in every continuation; probes do not reproduce original optimizer momentum'}
    (out/'design.json').write_text(json.dumps(facts,indent=2)+'\n');print(facts,flush=True)
    completed=(out/'complete.json').exists()
    perturb=pd.read_csv(out/'interventions.csv').to_dict('records') if completed else interventions(model,'best_checkpoint')
    if not completed:pd.DataFrame(perturb).to_csv(out/'interventions.csv',index=False)
    history=[];summaries=[]
    variants=[('continue',.001,.0001,False,0.),('lower_lr',.0002,.0001,False,0.),
              ('freeze_H',.001,.0001,True,0.),('weight_decay',.001,.01,False,0.),
              ('background_noise',.001,.0001,False,.2)]
    if completed:variants=[]
    for name,lr,wd,freeze,noise in variants:
        model.load_state_dict(state);model.raw_dictionary.requires_grad_(not freeze);expression.copy_(original)
        config=replace(cfg,lr=lr,weight_decay=wd)
        # Match the actual production fit's dimensionless amplitude optimizer.
        scale=payload['activity_parameter_scale'];head=model.activity_head
        with torch.no_grad():head.weight.div_(scale);head.bias.div_(scale)
        amplitude_hook=head.register_forward_hook(lambda module,args,value:value*scale)
        noise_hook=None
        if noise:
            mask=torch.tensor(non_lr,device=device,dtype=expression.dtype)
            def augment(module,args):
                return (args[0]+torch.randn_like(args[0])*mask*noise,) if module.training else args
            noise_hook=model.cell_encoder.register_forward_pre_hook(augment)
        optimizer=make_optimizer(model,config);set_seed(cfg.seed)
        start=evaluate(model,val);best=start;best_e=0
        for e in range(13):
            if e:epoch(model,data,fit,config,optimizer,e,objective)
            tr=evaluate(model,monitor);sm=evaluate(model,same);va=evaluate(model,val)
            if va<best:best=va;best_e=e
            h=model.dictionary.detach().cpu().numpy()
            history.append({'variant':name,'extra_epoch':e,'lr':lr,'weight_decay':wd,'train_CCC':tr,
                            'same_slice_control_CCC':sm,'validation_CCC':va,'H_relative_change':float(np.linalg.norm(h-initial_h)/np.linalg.norm(initial_h))})
            pd.DataFrame(history).to_csv(out/'continuation_history.csv',index=False)
            print(name,e,f'train {tr:.5f} same-slice {sm:.5f} new-slice {va:.5f}',flush=True)
        if noise_hook:noise_hook.remove()
        amplitude_hook.remove()
        with torch.no_grad():head.weight.mul_(scale);head.bias.mul_(scale)
        if name=='continue':
            perturb.extend(interventions(model,'continued_checkpoint'))
            pd.DataFrame(perturb).to_csv(out/'interventions.csv',index=False)
        summaries.append({'variant':name,'starting_validation':start,'best_validation':best,'best_extra_epoch':best_e,
                          'last_validation':va,'last_train':tr,'last_same_slice':sm})
        pd.DataFrame(summaries).to_csv(out/'continuation_summary.csv',index=False)
    # Fresh-start controls distinguish input shortcuts from network width.
    # All three use the same production NMF dictionary/prior; no truth enters.
    expression.copy_(original)
    template=initialize(data,cfg).to(device)
    np.testing.assert_allclose(template.dictionary.detach().cpu().numpy(),np.load(ROOT/'gnn/initial_H.npy'),atol=1e-6)
    template_state=copy.deepcopy(template.state_dict())
    compact=np.array([data.genes[i] in lr_genes or data.genes[i].startswith('SYN_MARKER_') for i in data.encoder_indices]+[True])
    panel=np.flatnonzero(~compact);capacity_history=[];capacity_summary=[]
    for name,width in [('fresh_all_genes',64),('fresh_LR_markers',64),('fresh_hidden32',32)]:
        expression.copy_(original)
        if name=='fresh_LR_markers':expression[:,torch.tensor(panel,device=device)]=0
        config=replace(cfg,hidden=width)
        set_seed(cfg.seed);m=model_from_config(data,config).to(device)
        if width==64:m.load_state_dict(template_state)
        else:
            with torch.no_grad():
                m.raw_dictionary.copy_(template.raw_dictionary)
                m.composition_head.weight.zero_();m.composition_head.bias.copy_(template.composition_head.bias)
                m.activity_head.weight.zero_();m.activity_head.bias.copy_(template.activity_head.bias)
        scale=max(float(m.activity_head.bias.detach().abs().max()),1.)
        with torch.no_grad():m.activity_head.weight.div_(scale);m.activity_head.bias.div_(scale)
        hook=m.activity_head.register_forward_hook(lambda module,args,value:value*scale)
        optimizer=make_optimizer(m,config);set_seed(cfg.seed);best=float('inf');best_e=0
        for e in range(1,25):
            epoch(m,data,fit,config,optimizer,e-1,objective)
            tr=evaluate(m,monitor);sm=evaluate(m,same);va=evaluate(m,val)
            if va<best:best=va;best_e=e
            capacity_history.append({'variant':name,'epoch':e,'optimizer_steps':e*int(np.ceil(len(fit)/cfg.batch_size)),
                                     'train_CCC':tr,'same_slice_CCC':sm,'validation_CCC':va})
            pd.DataFrame(capacity_history).to_csv(out/'capacity_history.csv',index=False)
            if e%4==0:print(name,e,f'train {tr:.5f} same-slice {sm:.5f} validation {va:.5f}',flush=True)
        capacity_summary.append({'variant':name,'hidden':width,'active_genes':int(compact[:-1].sum()) if name=='fresh_LR_markers' else len(data.genes),
                                 'best_validation':best,'best_epoch':best_e,'last_validation':va,'last_train':tr,
                                 'last_same_slice':sm,'parameters':sum(p.numel() for p in m.parameters())})
        pd.DataFrame(capacity_summary).to_csv(out/'capacity_summary.csv',index=False)
        hook.remove();del m
    expression.copy_(original)
    assert all(checksum(ROOT/f)==digest for f,digest in sources.items())
    (out/'complete.json').write_text(json.dumps({'original_results_unchanged':True,'source_hashes':sources,'test_slices_used':False},indent=2)+'\n')


if __name__=='__main__':main()
