"""Spatially separated training and streamed inference, independent of the legacy AE."""
from pathlib import Path
import copy
import json
import math
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import MiniBatchKMeans
from ..training import set_seed
from ..data import resolve_device
from .biology import BioAlignment
from .model import MicroenvironmentGNN
from .losses import ccc_loss,regularization,guarded_backward,contrastive_loss


def initialize(data,config):
    set_seed(config.seed)
    model=MicroenvironmentGNN(len(data.encoder_indices)+1,len(data.categories),len(data.features),config.programs,config.hidden,config.sigma_um,data.feature_mask)
    sample=np.random.default_rng(config.seed).choice(data.rows[0],min(len(data.rows[0]),4096),replace=False)
    if len(sample)<config.programs: raise ValueError('Too few independent training centers for program dictionary')
    target=data.targets[sample]
    centers=MiniBatchKMeans(n_clusters=config.programs,random_state=config.seed,n_init=3,batch_size=512).fit(target/data.feature_scale).cluster_centers_ * data.feature_scale
    centers=np.maximum(centers,1e-7)*data.feature_mask
    centers/=np.maximum(centers.sum(1,keepdims=True),1e-8)
    with torch.no_grad():
        model.raw_dictionary.copy_(torch.from_numpy(np.log(np.expm1(np.maximum(centers,1e-7)))))
        activity=max(float(np.median(target[:,data.feature_mask].sum(1))),1e-4)
        bias=activity if activity>20 else math.log(math.expm1(activity))
        model.activity_head.weight.zero_();model.activity_head.bias.fill_(bias)
    return model


def fit_alignments(data,config):
    sample=np.sort(np.random.default_rng(config.split_seed).choice(data.rows[0],min(len(data.rows[0]),20000),replace=False))
    alignments={}
    for branch in ('upstream','downstream'):
        if not getattr(config,'use_'+branch): continue
        if branch not in data.observations: raise ValueError(f'{branch} has no verified observed supervision')
        alignment=BioAlignment.fit(data.priors[branch],data.targets[sample],data.nuisance[sample],data.observations[branch][sample],data.observed_masks[branch][sample],np.arange(len(sample)))
        if not alignment.valid.any(): raise ValueError(f'{branch}: no nonconstant training regulatory alignment')
        alignments[branch]=alignment
    return alignments


def epoch(model,data,rows,config,alignments,optimizer=None,epoch_number=0):
    training=optimizer is not None;model.train(training)
    device=next(model.parameters()).device
    rng=np.random.default_rng(config.seed+epoch_number);rows=rng.permutation(rows) if training else rows
    sums={};count=0
    with torch.set_grad_enabled(training):
        for start in range(0,len(rows),config.batch_size):
            chosen=rows[start:start+config.batch_size];batch=data.batch(chosen,device)
            result=data.forward(model,batch)
            ccc=ccc_loss(result['reconstruction'],batch['target'],data.feature_scale,data.feature_mask)
            reg=regularization(result)
            bio=result['Z'].sum()*0; metrics={'ccc':float(ccc.detach()),'regularization':float(reg.detach())}
            for branch,alignment in alignments.items():
                value=alignment.loss(result['reconstruction'],batch['nuisance'],batch[branch],batch[branch+'_mask'])
                metrics[branch]=float(value.detach());bio=bio+(config.alpha if branch=='upstream' else config.beta)*value
            ramp=min(1.,(epoch_number+1)/max(config.bio_warmup_epochs,1))
            auxiliary=config.lambda_reg*reg+config.lambda_bio*ramp*bio
            if config.contrastive:
                contrast,valid=contrastive_loss(data,model,batch,result,data.feature_scale)
                auxiliary=auxiliary+config.lambda_contrast*contrast
                metrics['contrast']=float(contrast.detach());metrics['contrast_valid']=valid/len(chosen)
            primary=ccc
            metrics['total']=float((primary+auxiliary).detach())
            if training:
                optimizer.zero_grad(set_to_none=True)
                metrics.update(guarded_backward(primary,auxiliary,model.parameters(),config.gradient_clip))
                optimizer.step()
            for key,value in metrics.items(): sums[key]=sums.get(key,0)+value*len(chosen)
            count+=len(chosen)
    return {key:value/count for key,value in sums.items()}


def fit(data,config):
    output=Path(config.output_dir);output.mkdir(parents=True,exist_ok=True)
    if (output/'checkpoint.pt').exists(): raise FileExistsError('GNN checkpoint exists; use inference or a distinct experiment output')
    device=resolve_device(config.device);model=initialize(data,config).to(device)
    alignments=fit_alignments(data,config)
    if not config.contrastive and config.prior_control=='none': data.prepare_device(device,output/'cache')
    optimizer=torch.optim.AdamW(model.parameters(),lr=config.lr,weight_decay=config.weight_decay)
    history=[];best=float('inf');state=None;stale=0
    for e in range(config.max_epochs):
        print(f'Epoch {e+1}/{config.max_epochs}',flush=True)
        tr=epoch(model,data,data.rows[0],config,alignments,optimizer,e)
        va=epoch(model,data,data.rows[1],config,alignments,None,e)
        # The warmup weight is not used to compare early-stopping scores across epochs.
        score=va['ccc']+config.lambda_reg*va['regularization']+config.lambda_bio*(config.alpha*va.get('upstream',0)+config.beta*va.get('downstream',0))
        print(f'Epoch {e+1}: train CCC={tr["ccc"]:.6f}, validation CCC={va["ccc"]:.6f}, selected loss={score:.6f}',flush=True)
        history.append({'epoch':e+1,**{'train_'+k:v for k,v in tr.items()},**{'validation_'+k:v for k,v in va.items()},'selection_loss':score})
        pd.DataFrame(history).to_csv(output/'training.csv',index=False)
        if score<best-1e-7:
            best=score;state=copy.deepcopy(model.state_dict());stale=0;best_epoch=e+1
            torch.save({'model_state':state,'epoch':best_epoch,'config':config.to_dict()},output/'training_best.pt')
        else:
            stale+=1
            if stale>=config.patience: break
    model.load_state_dict(state)
    payload={'model_state':state,'config':config.to_dict(),'genes':data.genes,'categories':data.categories,
             'encoder_indices':data.encoder_indices,'features':data.features.to_dict('list'),
             'gene_mean':data.gene_mean,'gene_scale':data.gene_scale,'feature_scale':data.feature_scale,
             'feature_mask':data.feature_mask,'folds':data.folds,'alignments':alignments,'best_epoch':best_epoch,'data_audit':data.audit}
    torch.save(payload,output/'checkpoint.pt')
    pd.DataFrame(history).to_csv(output/'training.csv',index=False)
    (output/'config.json').write_text(json.dumps(config.to_dict(),indent=2)+'\n')
    test=epoch(model,data,data.rows[2],config,alignments,None,config.max_epochs)
    (output/'heldout_metrics.json').write_text(json.dumps({'best_epoch':best_epoch,'validation_loss':best,'test':test},indent=2)+'\n')
    return model,alignments


def infer(model,data,output,batch_size=128):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);model.eval()
    paths={k:output/(k+'.npy') for k in ('Z','P','activity')}
    arrays={k:np.lib.format.open_memmap(p.with_suffix('.incomplete.npy'),mode='w+',dtype=np.float32,shape=((len(data.cells),model.programs) if k!='activity' else (len(data.cells),))) for k,p in paths.items()}
    with torch.no_grad():
        for start in range(0,len(data.cells),batch_size):
            rows=np.arange(start,min(start+batch_size,len(data.cells)))
            batch=data.batch(rows,next(model.parameters()).device,include_bio=False);result=data.forward(model,batch)
            for key in arrays: arrays[key][rows]=result[key].cpu().numpy()
    for key,p in paths.items(): arrays[key].flush();del arrays[key];p.with_suffix('.incomplete.npy').replace(p)
    np.save(output/'H.npy',model.dictionary.detach().cpu().numpy())
    return {key:np.load(p,mmap_mode='r') for key,p in paths.items()}
