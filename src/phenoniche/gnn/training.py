"""Spatially separated training and streamed inference, independent of the legacy AE."""
from pathlib import Path
import copy
import json
import math
import hashlib
import warnings
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning
from ..training import set_seed
from ..data import resolve_device
from .model import model_from_config
from .losses import ccc_loss,ccc_error_sums,regularization,guarded_backward,composition_loss
from functools import partial


def make_optimizer(model,config):
    if config.dictionary_parameterization=='softplus':
        return torch.optim.AdamW(model.parameters(),lr=config.lr,weight_decay=config.weight_decay)
    encoder=[p for p in model.parameters() if p is not model.raw_dictionary]
    return torch.optim.AdamW([{'params':encoder,'lr':config.lr,'weight_decay':config.weight_decay},
        {'params':[model.raw_dictionary],'lr':config.dictionary_lr,'weight_decay':0.}])


def initialize(data,config,initialize_composition=False):
    set_seed(config.seed)
    model=model_from_config(data,config)
    nmf=config.dictionary_initialization=='nmf'
    budget=config.nmf_initialization_sample_size if nmf else 4096
    sample=np.random.default_rng(config.seed).choice(data.rows[0],min(len(data.rows[0]),budget),replace=False)
    if len(sample)<config.programs: raise ValueError('Too few independent training centers for program dictionary')
    target=data.targets[sample]
    robust=config.dictionary_initialization=='log1p_active_kmeans'
    factor_activity=None
    if nmf:
        # Fit only training CCC on effective features. Undo RMS conditioning in
        # the dictionary, then transfer its row mass to Z to preserve C = ZH.
        conditioned=target[:,data.feature_mask]/data.feature_scale[data.feature_mask]
        if not np.isfinite(conditioned).all() or (conditioned<0).any() or not conditioned.any():
            raise ValueError('NMF requires finite nonnegative, nonempty training CCC')
        estimator=NMF(n_components=config.programs,init='nndsvda' if min(conditioned.shape)>=config.programs else 'random',
            random_state=config.seed,max_iter=config.nmf_initialization_max_iter,tol=1e-4)
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter('always',ConvergenceWarning)
            factor_activity=estimator.fit_transform(conditioned).astype(np.float32)
        native_h=estimator.components_*data.feature_scale[data.feature_mask][None]
        mass=native_h.sum(1)
        if not np.isfinite(mass).all() or (mass<=0).any():
            raise ValueError('NMF produced an empty/nonfinite program; reduce the program count or inspect training CCC')
        factor_activity*=mass[None]
        centers=np.zeros((config.programs,len(data.features)),np.float32)
        centers[:,data.feature_mask]=native_h/mass[:,None]
        assigned=factor_activity.argmax(1)
        model.initialization_audit={'method':'nmf','fit':'training spatial blocks only; no validation/test CCC',
            'training_centers':len(sample),'training_rows_sha256':hashlib.sha256(sample.astype('<i8').tobytes()).hexdigest(),
            'prototype_features':int(data.feature_mask.sum()),'feature_mask_sha256':hashlib.sha256(data.feature_mask.tobytes()).hexdigest(),
            'seed':config.seed,'iterations':int(estimator.n_iter_),'max_iter':config.nmf_initialization_max_iter,
            'converged':not any(issubclass(w.category,ConvergenceWarning) for w in recorded),
            'scaled_training_relative_error':float(estimator.reconstruction_err_/np.linalg.norm(conditioned)),
            'cluster_counts':np.bincount(assigned,minlength=config.programs).tolist(),
            'dictionary_definition':'RMS-conditioned NMF loading restored to native CCC units; unit row mass',
            'encoder_pretraining':False,'optimization':'joint CCC reconstruction + regularization'}
    else:
        prototype_input=np.log1p(target[:,data.feature_mask]/data.feature_scale[data.feature_mask]) if robust else target/data.feature_scale
        prototype=MiniBatchKMeans(n_clusters=config.programs,random_state=config.seed,n_init=3,batch_size=512).fit(prototype_input)
        assigned=prototype.predict(prototype_input)
    if robust:
        # Transform only distances used for initialization. H remains in native
        # CCC units, from actual training-cluster means, never inverse-log centroids.
        centers=np.stack([target[assigned==p].mean(0) if (assigned==p).any() else target.mean(0) for p in range(config.programs)])
    elif not nmf:centers=prototype.cluster_centers_ * data.feature_scale
    if not nmf:
        model.initialization_audit={'method':config.dictionary_initialization,'training_centers':len(sample),
            'prototype_features':int(data.feature_mask.sum()) if robust else target.shape[1],
            'cluster_counts':np.bincount(assigned,minlength=config.programs).tolist(),
            'native_centroid_definition':'observed training-cluster CCC means' if robust else 'RMS-scaled KMeans centroids'}
    centers=np.maximum(centers,1e-7)*data.feature_mask
    centers/=np.maximum(centers.sum(1,keepdims=True),1e-8)
    with torch.no_grad():
        coordinates=centers.copy()
        if config.dictionary_coordinates=='feature_scaled':
            coordinates/=data.feature_scale
            coordinates/=np.maximum(coordinates.sum(1,keepdims=True),1e-8)
        values=coordinates if config.dictionary_parameterization=='projected' else np.log(np.expm1(np.maximum(coordinates,1e-7)))
        model.raw_dictionary.copy_(torch.from_numpy(values))
        model.project_dictionary()
        activity=max(float(np.median(factor_activity.sum(1) if nmf else target[:,data.feature_mask].sum(1))),1e-4)
        bias=activity if activity>20 else math.log(math.expm1(activity))
        model.activity_head.weight.zero_();model.activity_head.bias.fill_(bias)
        if nmf:
            prior=factor_activity.mean(0);prior/=prior.sum()
            model.composition_head.weight.zero_();model.composition_head.bias.copy_(torch.from_numpy(np.log(np.maximum(prior,1e-8))))
        if config.activity_parameterization=='independent':
            values=activity*np.maximum(prior,1e-8) if nmf else np.full(config.programs,activity/config.programs)
            inverse=values+np.log(-np.expm1(-values))
            model.program_head.weight.zero_();model.program_head.bias.copy_(torch.as_tensor(inverse,dtype=model.program_head.bias.dtype))
        if initialize_composition:
            pi=np.asarray(data.composition_target[sample]);valid=pi.sum(1)>0
            prior=pi[valid].mean(0) if valid.any() else np.full(len(data.categories),1/len(data.categories))
            typical=np.stack([pi[(assigned==p)&valid].mean(0) if ((assigned==p)&valid).any() else prior for p in range(config.programs)])
            typical=np.maximum(typical,.001);typical/=typical.sum(1,keepdims=True)
            model.raw_composition.copy_(torch.from_numpy(np.log(typical)).float())
    return model


def epoch(model,data,rows,config,optimizer=None,epoch_number=0,ccc_objective=None):
    training=optimizer is not None;model.train(training)
    device=next(model.parameters()).device
    rng=np.random.default_rng(config.seed+epoch_number);rows=rng.permutation(rows) if training else rows
    sums={};count=0;composition_count=0;error_sums={}
    objective=ccc_objective or partial(ccc_loss,nonzero_fraction=data.feature_nonzero_fraction,nonzero_weight_max=config.ccc_nonzero_weight_max)
    with torch.set_grad_enabled(training):
        for start in range(0,len(rows),config.batch_size):
            chosen=rows[start:start+config.batch_size];batch=data.batch(chosen,device)
            result=data.forward(model,batch)
            batch_objective=partial(objective,batch=batch) if getattr(objective,'uses_batch',False) else objective
            ccc=batch_objective(result['reconstruction'],batch['target'],data.feature_scale,data.feature_mask)
            reg=regularization(result)
            comp=composition_loss(result['composition'],batch['composition_target']) if config.lambda_comp>0 else ccc.new_zeros(())
            metrics={'ccc':float(ccc.detach()),'regularization':float(reg.detach())}
            diagnostic=ccc_error_sums(result['reconstruction'],batch['target'],data.feature_scale,data.feature_mask)
            for key,value in diagnostic.items():error_sums[key]=error_sums.get(key,0)+value
            with torch.no_grad():
                metrics['ccc_zero_baseline']=float(batch_objective(torch.zeros_like(result['reconstruction']),batch['target'],data.feature_scale,data.feature_mask))
                if data.feature_mean is not None:
                    mean=torch.as_tensor(data.feature_mean,device=device,dtype=batch['target'].dtype).expand_as(batch['target'])
                    metrics['ccc_mean_baseline']=float(batch_objective(mean,batch['target'],data.feature_scale,data.feature_mask))
            if config.lambda_comp>0:metrics['composition']=float(comp.detach())
            auxiliary=config.lambda_reg*reg+config.lambda_comp*comp
            primary=ccc
            metrics['total']=float((primary+auxiliary).detach())
            if training:
                optimizer.zero_grad(set_to_none=True)
                metrics.update(guarded_backward(primary,auxiliary,model.parameters(),config.gradient_clip))
                optimizer.step()
                model.project_dictionary()
            valid_count=int((batch['composition_target'].sum(1)>0).sum()) if config.lambda_comp>0 else 0
            composition_count+=valid_count
            for key,value in metrics.items(): sums[key]=sums.get(key,0)+value*(valid_count if key=='composition' else len(chosen))
            count+=len(chosen)
    result={key:value/(max(composition_count,1) if key=='composition' else count) for key,value in sums.items()}
    result['ccc_unweighted']=error_sums['huber_sum']/max(error_sums['entries'],1)
    result['ccc_positive_huber']=error_sums['positive_huber_sum']/max(error_sums['positive_entries'],1)
    result['ccc_zero_huber']=error_sums['zero_huber_sum']/max(error_sums['zero_entries'],1)
    result['ccc_positive_fraction']=error_sums['positive_entries']/max(error_sums['entries'],1)
    result['ccc_relative_RMSE']=math.sqrt(error_sums['native_sse']/max(error_sums['native_target_square'],1e-12))
    result['ccc_gain_over_zero']=1-result['ccc']/max(result['ccc_zero_baseline'],1e-12)
    if 'ccc_mean_baseline' in result:result['ccc_gain_over_mean']=1-result['ccc']/max(result['ccc_mean_baseline'],1e-12)
    if config.lambda_comp>0:result['composition_valid_centers']=composition_count
    result['total']=result['ccc']+config.lambda_reg*result['regularization']+config.lambda_comp*result.get('composition',0.)
    return result


def fit(data,config,initialize_composition=False,rescale_activity=False,ccc_objective=None):
    output=Path(config.output_dir);output.mkdir(parents=True,exist_ok=True)
    if (output/'checkpoint.pt').exists(): raise FileExistsError('GNN checkpoint exists; use inference or a distinct experiment output')
    if data.feature_mean is None or data.feature_nonzero_fraction is None:
        total=np.zeros(len(data.features));nonzero=np.zeros_like(total)
        for start in range(0,len(data.rows[0]),4096):
            block=data.targets[data.rows[0][start:start+4096]];total+=block.sum(0);nonzero+=(block>0).sum(0)
        data.feature_mean=total/len(data.rows[0]);data.feature_nonzero_fraction=nonzero/len(data.rows[0])
    device=resolve_device(config.device);model=initialize(data,config,initialize_composition).to(device)
    (output/'initialization_audit.json').write_text(json.dumps(model.initialization_audit,indent=2)+'\n')
    # Optimizer-only reparameterization: learn the large native activity in
    # dimensionless units. Fuse the constant into Linear weights before
    # checkpoint/export, keeping the original model/inference architecture.
    amplitude_head=model.program_head if config.activity_parameterization=='independent' else model.activity_head
    amplitude_prefix='program_head' if config.activity_parameterization=='independent' else 'activity_head'
    activity_parameter_scale=max(float(amplitude_head.bias.detach().abs().max()),1.) if rescale_activity else 1.
    activity_hook=None
    if rescale_activity:
        with torch.no_grad():
            amplitude_head.weight.div_(activity_parameter_scale)
            amplitude_head.bias.div_(activity_parameter_scale)
        activity_hook=amplitude_head.register_forward_hook(lambda module,args,value:value*activity_parameter_scale)
    if initialize_composition or rescale_activity or config.dictionary_initialization=='nmf':
        np.save(output/'initial_H.npy',model.dictionary.detach().cpu().numpy())
    if initialize_composition or rescale_activity:
        np.save(output/'initial_B.npy',model.composition_dictionary.detach().cpu().numpy())
    data.prepare_device(device,output/'cache')
    optimizer=make_optimizer(model,config)
    monitor_role=data.audit.get('monitor_role','validation')
    monitor_name='training monitor' if monitor_role=='training_subset' else 'validation'
    history=[];best=float('inf');state=None;stale=0
    for e in range(config.max_epochs):
        print(f'Epoch {e+1}/{config.max_epochs}',flush=True)
        tr=epoch(model,data,data.rows[0],config,optimizer,e,ccc_objective)
        va=epoch(model,data,data.rows[1],config,None,e,ccc_objective)
        # Compare the same CCC-primary objective across all epochs.
        score=va['ccc']+config.lambda_reg*va['regularization']+config.lambda_comp*va.get('composition',0.)
        print(f'Epoch {e+1}: train CCC={tr["ccc"]:.6f}, {monitor_name} CCC={va["ccc"]:.6f}, selected loss={score:.6f}; {monitor_name} unweighted={va["ccc_unweighted"]:.6f}, positive={va["ccc_positive_huber"]:.6f}, gain vs zero={va["ccc_gain_over_zero"]:.1%}',flush=True)
        prefix='monitor_' if monitor_role=='training_subset' else 'validation_'
        history.append({'epoch':e+1,**{'train_'+k:v for k,v in tr.items()},**{prefix+k:v for k,v in va.items()},'selection_loss':score})
        pd.DataFrame(history).to_csv(output/'training.csv',index=False)
        if score<best-1e-7:
            best=score;state=copy.deepcopy(model.state_dict());stale=0;best_epoch=e+1
            native_state=copy.deepcopy(state)
            for key in (amplitude_prefix+'.weight',amplitude_prefix+'.bias'):native_state[key]*=activity_parameter_scale
            torch.save({'model_state':native_state,'epoch':best_epoch,'config':config.to_dict(),'activity_parameter_scale':activity_parameter_scale},output/'training_best.pt')
        else:
            stale+=1
            if stale>=config.patience: break
    model.load_state_dict(state)
    if activity_hook is not None:
        activity_hook.remove()
        with torch.no_grad():
            amplitude_head.weight.mul_(activity_parameter_scale)
            amplitude_head.bias.mul_(activity_parameter_scale)
    state=copy.deepcopy(model.state_dict())
    payload={'model_state':state,'config':config.to_dict(),'genes':data.genes,'categories':data.categories,
             'encoder_indices':data.encoder_indices,'features':data.features.to_dict('list'),
             'gene_mean':data.gene_mean,'gene_scale':data.gene_scale,'feature_scale':data.feature_scale,
             'feature_mask':data.feature_mask,'feature_mean':data.feature_mean,'feature_nonzero_fraction':data.feature_nonzero_fraction,'folds':data.folds,'architecture':config.architecture,'best_epoch':best_epoch,'data_audit':data.audit,'activity_parameter_scale':activity_parameter_scale}
    payload.update(expression_normalization=data.expression_normalization,library_mean=data.library_mean,library_scale=data.library_scale)
    if data.cell_state_preprocessing is not None:payload['cell_state_preprocessing']=data.cell_state_preprocessing
    torch.save(payload,output/'checkpoint.pt')
    pd.DataFrame(history).to_csv(output/'training.csv',index=False)
    (output/'config.json').write_text(json.dumps(config.to_dict(),indent=2)+'\n')
    if monitor_role=='training_subset':
        (output/'training_monitor_metrics.json').write_text(json.dumps({'best_epoch':best_epoch,'monitor_loss':best,
            'monitor_role':monitor_role,'test':None,'training_centers':len(data.rows[0]),'monitor_centers':len(data.rows[1])},indent=2)+'\n')
    else:
        test=epoch(model,data,data.rows[2],config,None,config.max_epochs,ccc_objective) if len(data.rows[2]) else None
        (output/'heldout_metrics.json').write_text(json.dumps({'best_epoch':best_epoch,'validation_loss':best,'test':test},indent=2)+'\n')
    return model


def infer(model,data,output,batch_size=128):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);model.eval()
    paths={k:output/(k+'.npy') for k in ('Z','P','activity')}
    arrays={k:np.lib.format.open_memmap(p.with_suffix('.incomplete.npy'),mode='w+',dtype=np.float32,shape=((len(data.cells),model.programs) if k!='activity' else (len(data.cells),))) for k,p in paths.items()}
    with torch.no_grad():
        for start in range(0,len(data.cells),batch_size):
            rows=np.arange(start,min(start+batch_size,len(data.cells)))
            batch=data.batch(rows,next(model.parameters()).device,include_target=False);result=data.forward(model,batch,reconstruct=False)
            for key in arrays: arrays[key][rows]=result[key].cpu().numpy()
    for key,p in paths.items(): arrays[key].flush();del arrays[key];p.with_suffix('.incomplete.npy').replace(p)
    np.save(output/'H.npy',model.dictionary.detach().cpu().numpy())
    np.save(output/'B.npy',model.composition_dictionary.detach().cpu().numpy())
    return {key:np.load(p,mmap_mode='r') for key,p in paths.items()}
