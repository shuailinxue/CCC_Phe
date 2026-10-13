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
from .model import MicroenvironmentGNN
from .losses import ccc_loss,regularization,guarded_backward,composition_loss


def initialize(data,config,initialize_composition=False):
    set_seed(config.seed)
    model=MicroenvironmentGNN(len(data.encoder_indices)+1,len(data.categories),len(data.features),config.programs,config.hidden,config.sigma_um,data.feature_mask)
    sample=np.random.default_rng(config.seed).choice(data.rows[0],min(len(data.rows[0]),4096),replace=False)
    if len(sample)<config.programs: raise ValueError('Too few independent training centers for program dictionary')
    target=data.targets[sample]
    prototype=MiniBatchKMeans(n_clusters=config.programs,random_state=config.seed,n_init=3,batch_size=512).fit(target/data.feature_scale)
    centers=prototype.cluster_centers_ * data.feature_scale
    centers=np.maximum(centers,1e-7)*data.feature_mask
    centers/=np.maximum(centers.sum(1,keepdims=True),1e-8)
    with torch.no_grad():
        model.raw_dictionary.copy_(torch.from_numpy(np.log(np.expm1(np.maximum(centers,1e-7)))))
        activity=max(float(np.median(target[:,data.feature_mask].sum(1))),1e-4)
        bias=activity if activity>20 else math.log(math.expm1(activity))
        model.activity_head.weight.zero_();model.activity_head.bias.fill_(bias)
        if initialize_composition:
            pi=np.asarray(data.composition_target[sample]);valid=pi.sum(1)>0
            prior=pi[valid].mean(0) if valid.any() else np.full(len(data.categories),1/len(data.categories))
            assigned=prototype.predict(target/data.feature_scale)
            typical=np.stack([pi[(assigned==p)&valid].mean(0) if ((assigned==p)&valid).any() else prior for p in range(config.programs)])
            typical=np.maximum(typical,.001);typical/=typical.sum(1,keepdims=True)
            model.raw_composition.copy_(torch.from_numpy(np.log(typical)).float())
    return model


def epoch(model,data,rows,config,optimizer=None,epoch_number=0,ccc_objective=None):
    training=optimizer is not None;model.train(training)
    device=next(model.parameters()).device
    rng=np.random.default_rng(config.seed+epoch_number);rows=rng.permutation(rows) if training else rows
    sums={};count=0;composition_count=0
    with torch.set_grad_enabled(training):
        for start in range(0,len(rows),config.batch_size):
            chosen=rows[start:start+config.batch_size];batch=data.batch(chosen,device)
            result=data.forward(model,batch)
            ccc=(ccc_objective or ccc_loss)(result['reconstruction'],batch['target'],data.feature_scale,data.feature_mask)
            reg=regularization(result)
            comp=composition_loss(result['composition'],batch['composition_target']) if config.lambda_comp>0 else ccc.new_zeros(())
            metrics={'ccc':float(ccc.detach()),'regularization':float(reg.detach())}
            if config.lambda_comp>0:metrics['composition']=float(comp.detach())
            auxiliary=config.lambda_reg*reg+config.lambda_comp*comp
            primary=ccc
            metrics['total']=float((primary+auxiliary).detach())
            if training:
                optimizer.zero_grad(set_to_none=True)
                metrics.update(guarded_backward(primary,auxiliary,model.parameters(),config.gradient_clip))
                optimizer.step()
            valid_count=int((batch['composition_target'].sum(1)>0).sum()) if config.lambda_comp>0 else 0
            composition_count+=valid_count
            for key,value in metrics.items(): sums[key]=sums.get(key,0)+value*(valid_count if key=='composition' else len(chosen))
            count+=len(chosen)
    result={key:value/(max(composition_count,1) if key=='composition' else count) for key,value in sums.items()}
    if config.lambda_comp>0:result['composition_valid_centers']=composition_count
    result['total']=result['ccc']+config.lambda_reg*result['regularization']+config.lambda_comp*result.get('composition',0.)
    return result


def fit(data,config,initialize_composition=False,rescale_activity=False,ccc_objective=None):
    output=Path(config.output_dir);output.mkdir(parents=True,exist_ok=True)
    if (output/'checkpoint.pt').exists(): raise FileExistsError('GNN checkpoint exists; use inference or a distinct experiment output')
    device=resolve_device(config.device);model=initialize(data,config,initialize_composition).to(device)
    # Optimizer-only reparameterization: learn the large native activity in
    # dimensionless units. Fuse the constant into Linear weights before
    # checkpoint/export, keeping the original model/inference architecture.
    activity_parameter_scale=max(float(model.activity_head.bias.detach().abs()),1.) if rescale_activity else 1.
    activity_hook=None
    if rescale_activity:
        with torch.no_grad():
            model.activity_head.weight.div_(activity_parameter_scale)
            model.activity_head.bias.div_(activity_parameter_scale)
        activity_hook=model.activity_head.register_forward_hook(lambda module,args,value:value*activity_parameter_scale)
    if initialize_composition or rescale_activity:
        np.save(output/'initial_H.npy',model.dictionary.detach().cpu().numpy())
        np.save(output/'initial_B.npy',model.composition_dictionary.detach().cpu().numpy())
    data.prepare_device(device,output/'cache')
    optimizer=torch.optim.AdamW(model.parameters(),lr=config.lr,weight_decay=config.weight_decay)
    history=[];best=float('inf');state=None;stale=0
    for e in range(config.max_epochs):
        print(f'Epoch {e+1}/{config.max_epochs}',flush=True)
        tr=epoch(model,data,data.rows[0],config,optimizer,e,ccc_objective)
        va=epoch(model,data,data.rows[1],config,None,e,ccc_objective)
        # Compare the same CCC-primary objective across all epochs.
        score=va['ccc']+config.lambda_reg*va['regularization']+config.lambda_comp*va.get('composition',0.)
        print(f'Epoch {e+1}: train CCC={tr["ccc"]:.6f}, validation CCC={va["ccc"]:.6f}, selected loss={score:.6f}',flush=True)
        history.append({'epoch':e+1,**{'train_'+k:v for k,v in tr.items()},**{'validation_'+k:v for k,v in va.items()},'selection_loss':score})
        pd.DataFrame(history).to_csv(output/'training.csv',index=False)
        if score<best-1e-7:
            best=score;state=copy.deepcopy(model.state_dict());stale=0;best_epoch=e+1
            native_state=copy.deepcopy(state)
            for key in ('activity_head.weight','activity_head.bias'):native_state[key]*=activity_parameter_scale
            torch.save({'model_state':native_state,'epoch':best_epoch,'config':config.to_dict(),'activity_parameter_scale':activity_parameter_scale},output/'training_best.pt')
        else:
            stale+=1
            if stale>=config.patience: break
    model.load_state_dict(state)
    if activity_hook is not None:
        activity_hook.remove()
        with torch.no_grad():
            model.activity_head.weight.mul_(activity_parameter_scale)
            model.activity_head.bias.mul_(activity_parameter_scale)
    state=copy.deepcopy(model.state_dict())
    payload={'model_state':state,'config':config.to_dict(),'genes':data.genes,'categories':data.categories,
             'encoder_indices':data.encoder_indices,'features':data.features.to_dict('list'),
             'gene_mean':data.gene_mean,'gene_scale':data.gene_scale,'feature_scale':data.feature_scale,
             'feature_mask':data.feature_mask,'folds':data.folds,'architecture':config.architecture,'best_epoch':best_epoch,'data_audit':data.audit,'activity_parameter_scale':activity_parameter_scale}
    torch.save(payload,output/'checkpoint.pt')
    pd.DataFrame(history).to_csv(output/'training.csv',index=False)
    (output/'config.json').write_text(json.dumps(config.to_dict(),indent=2)+'\n')
    test=epoch(model,data,data.rows[2],config,None,config.max_epochs,ccc_objective)
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
