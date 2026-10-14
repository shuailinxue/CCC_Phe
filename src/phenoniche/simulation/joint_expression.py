"""CCC-priority joint expression training; clinical analysis is never involved."""
from pathlib import Path
from dataclasses import replace
from functools import partial
import copy,json,shutil
import numpy as np
import pandas as pd
import anndata as ad
import torch
from ..storage import STORAGE_ROOT
from ..training import set_seed
from ..data import resolve_device
from ..gnn.pipeline import load_checkpoint
from ..gnn.training import infer
from ..gnn.losses import ccc_loss,regularization
from ..gnn.evaluation import discover
from .config import SimulationConfig
from .adapter import adapt,gnn_config
from .baselines import fixed_clustering,dictionary_program_projection
from .expression_decoder import FrozenProgramExpressionDecoder
from .response_simulation import checksum


def protected_gradients(primary,auxiliary,shared,ccc_parameters,cap=.1):
    """Asymmetric projection/cap on the encoder; H receives primary gradients only.

    This is a first-order safeguard, not a guarantee under Adam/finite steps.
    Validation CCC supplies the independent checkpoint acceptance test.
    """
    shared=list(shared);ccc_parameters=list(ccc_parameters)
    pg=torch.autograd.grad(primary,shared+ccc_parameters,retain_graph=True,allow_unused=True)
    ag=torch.autograd.grad(auxiliary,shared,retain_graph=True,allow_unused=True)
    p=[torch.zeros_like(v) if g is None else g for v,g in zip(shared,pg)]
    a=[torch.zeros_like(v) if g is None else g for v,g in zip(shared,ag)]
    norm=sum((g*g).sum() for g in p);dot=sum((g*h).sum() for g,h in zip(p,a))
    correction=dot.clamp_max(0)/norm.clamp_min(1e-20)
    a=[g-correction*h for g,h in zip(a,p)]
    anorm=sum((g*g).sum() for g in a).sqrt()
    factor=(cap*norm.sqrt()/anorm.clamp_min(1e-20)).clamp(max=1)
    for value,g,h in zip(shared,p,a):value.grad=g+factor*h
    for value,g in zip(ccc_parameters,pg[len(shared):]):value.grad=g
    return {'auxiliary_conflict':float(dot<0),'auxiliary_norm_ratio':float(anorm*factor/norm.sqrt().clamp_min(1e-20))}


def run_joint_experiment(config,base_config=None):
    from .response_experiment import FROZEN,metrics,loading_evaluation,run_expression_experiment
    base=base_config or SimulationConfig.from_json(config.base_simulation_config)
    source=Path(base.output_dir);control=Path(config.frozen_control_dir);out=Path(config.output_dir)
    if out.is_symlink() or not out.resolve().is_relative_to(STORAGE_ROOT.resolve()):raise ValueError('Joint outputs require direct NAS workspace paths')
    for previous in (source,control):
        if out.resolve().is_relative_to(previous.resolve()) or previous.resolve().is_relative_to(out.resolve()):raise ValueError('Joint output overlaps its controls')
    if not (control/'complete.json').exists():
        run_expression_experiment(replace(config,training_mode='frozen',output_dir=str(control)),base)
    # Reuse the EXACT observed data, labels and train-only transforms of the control.
    old=json.loads((control/'config.json').read_text())
    if old['base_config']!=base.to_dict():raise ValueError('Frozen control belongs to a different CCC simulation')
    for key in ('response_genes_per_program','negative_control_genes','response_log_fold','nuisance_sd','seed'):
        if old['expression_config'][key]!=getattr(config,key):raise ValueError('Response simulation changed: '+key)
    snapshots={name:checksum(source/name) for name in FROZEN}
    if snapshots!=json.loads((control/'frozen_anchor.json').read_text()):raise ValueError('CCC anchor changed since frozen control')
    out.mkdir(parents=True,exist_ok=True)
    inputs=('response_injected.h5ad','response_gene_design.csv','expression_target.npy','expression_present.npy','expression_normalization.npy','expression_split_rows.npz','ground_truth/receiver_exposure.npy','ground_truth/response_baseline_counts.npy')
    manifest={'expression_config':config.to_dict(),'base_config':base.to_dict(),'frozen_anchor':snapshots,
        'control_files':{name:checksum(control/name) for name in (*inputs,'expression/checkpoint.pt','expression/preprocessing.npz')},
        'implementation':{name:checksum(Path(__file__).with_name(name)) for name in ('joint_expression.py','expression_decoder.py')}}
    cfg=out/'config.json'
    if cfg.exists() and json.loads(cfg.read_text())!=manifest:raise ValueError('Joint configuration/source changed; preserve existing results in another NAS output')
    cfg.write_text(json.dumps(manifest,indent=2)+'\n')
    if (out/'complete.json').exists() and json.loads((out/'complete.json').read_text()).get('manifest')==manifest:
        required=('summary.json','gnn/checkpoint.pt','gnn/Z.npy','gnn/H.npy','gnn/CCC_reconstruction.npy','expression/prediction.npy','raw_clusters.csv','metrics.csv','joint_training.csv','niche_top10_CCC.csv','niche_cell_composition.csv')
        if all((out/p).is_file() for p in required):return out
    for name in inputs:
        dst=out/name;dst.parent.mkdir(parents=True,exist_ok=True)
        if not dst.exists():shutil.copy2(control/name,dst)
        if checksum(dst)!=manifest['control_files'][name]:raise ValueError('Joint/control input differs: '+name)
    adata=ad.read_h5ad(source/'injected.h5ad');enriched=ad.read_h5ad(out/'response_injected.h5ad')
    adapter_dir=out/'anchor_inputs';(adapter_dir/'input').mkdir(parents=True,exist_ok=True)
    cached=adapter_dir/'input/ccc.npy'
    if not cached.exists():shutil.copy2(source/'input/ccc.npy',cached)
    data=adapt(adata,pd.read_csv(source/'lr.csv'),base,adapter_dir)
    if checksum(cached)!=snapshots['input/ccc.npy']:raise ValueError('CCC target changed')
    gcfg=gnn_config(base,source/'gnn');model,payload=load_checkpoint(data,gcfg)
    device=resolve_device(config.device);model.to(device);data.prepare_device(device,out/'cache')
    pre=np.load(control/'expression/preprocessing.npz');keys=('program_mean','program_scale','baseline','feature_scale','feature_mask')
    decoder=FrozenProgramExpressionDecoder(*(pre[k] for k in keys),detach_programs=False).to(device)
    decoder.load_state_dict(torch.load(control/'expression/checkpoint.pt',map_location=device,weights_only=False)['model_state'])
    target=np.load(out/'expression_target.npy');present=np.load(out/'expression_present.npy');norm=np.load(out/'expression_normalization.npy')
    tensors=[torch.as_tensor(v,device=device) for v in (target,present,norm)]
    expression_rows=np.load(out/'expression_split_rows.npz');erows=[expression_rows[k] for k in ('train','validation','test')]
    eligible=np.zeros(len(adata),bool);eligible[erows[0]]=True
    objective=partial(ccc_loss,kind=base.ccc_error,delta=base.ccc_huber_delta)
    # Keep the source amplitude's dimensionless optimization, fusing it at export.
    amplitude=model.activity_head;amplitude_scale=max(float(payload.get('activity_parameter_scale',1.)),1.)
    with torch.no_grad():amplitude.weight.div_(amplitude_scale);amplitude.bias.div_(amplitude_scale)
    hook=amplitude.register_forward_hook(lambda module,args,value:value*amplitude_scale)
    shared=[p for n,p in model.named_parameters() if n not in ('raw_dictionary','raw_composition')]
    hparams=[model.raw_dictionary]
    optimizer=torch.optim.AdamW(shared+hparams,lr=config.joint_lr,weight_decay=gcfg.weight_decay)
    gene_optimizer=torch.optim.Adam(decoder.parameters(),lr=config.lr)
    def measure(rows,gene_rows):
        model.eval();decoder.eval();ccc=0.;gene=0.;entries=0
        with torch.no_grad():
            for start in range(0,len(rows),config.batch_size):
                r=rows[start:start+config.batch_size];batch=data.batch(r,device);result=data.forward(model,batch)
                ccc+=float(objective(result['reconstruction'],batch['target'],data.feature_scale,data.feature_mask))*len(r)
            for start in range(0,len(gene_rows),config.batch_size):
                r=gene_rows[start:start+config.batch_size];result=data.forward(model,data.batch(r,device,include_target=False),reconstruct=False)
                loss,count=decoder.loss(result['Z'],*(v[r] for v in tensors));gene+=float(loss)*count;entries+=count
        return ccc/len(rows),gene/max(entries,1)
    set_seed(config.seed)
    checkpoint=out/'joint_checkpoint.pt'
    initial_ccc,initial_gene=measure(data.rows[1],erows[1]);limit=initial_ccc*(1+config.ccc_validation_tolerance)
    if checkpoint.exists():
        saved=torch.load(checkpoint,map_location=device,weights_only=False)
        model.load_state_dict(saved['gnn_state']);decoder.load_state_dict(saved['gene_state']);best_epoch=saved['best_epoch']
        history=pd.read_csv(out/'joint_training.csv').to_dict('records')
    else:
        best_gene=initial_gene;best_epoch=0;stale=0
        best={'gnn_state':copy.deepcopy(model.state_dict()),'gene_state':copy.deepcopy(decoder.state_dict())}
        history=[{'epoch':0,'validation_ccc':initial_ccc,'validation_gene':initial_gene,'CCC_eligible':True}]
        for e in range(config.joint_epochs):
            model.train();decoder.train();total=0.;rna=0.;count=0;rnacount=0;conflicts=[];ratios=[]
            rows=np.random.default_rng(config.seed+e).permutation(data.rows[0])
            for start in range(0,len(rows),config.batch_size):
                r=rows[start:start+config.batch_size];batch=data.batch(r,device);result=data.forward(model,batch)
                ccc=objective(result['reconstruction'],batch['target'],data.feature_scale,data.feature_mask)
                primary=ccc+base.lambda_reg*regularization(result)
                keep=np.flatnonzero(eligible[r]);optimizer.zero_grad(set_to_none=True);gene_optimizer.zero_grad(set_to_none=True)
                if len(keep):
                    gene,n=decoder.loss(result['Z'][keep],*(v[r[keep]] for v in tensors))
                    guards=protected_gradients(primary,config.lambda_gene*gene,shared,hparams,config.auxiliary_gradient_cap)
                    # Decoder task scaling is absorbed into its own learning rate;
                    # the auxiliary weight governs the competing shared gradient.
                    dg=torch.autograd.grad(gene+config.lambda_dictionary*decoder.dictionary.abs().mean(),decoder.parameters())
                    for p,g in zip(decoder.parameters(),dg):p.grad=g
                    torch.nn.utils.clip_grad_norm_(decoder.parameters(),5.);gene_optimizer.step()
                    rna+=float(gene.detach())*n;rnacount+=n;conflicts.append(guards['auxiliary_conflict']);ratios.append(guards['auxiliary_norm_ratio'])
                else:primary.backward()
                torch.nn.utils.clip_grad_norm_(shared+hparams,5.);optimizer.step();model.project_dictionary()
                total+=float(ccc.detach())*len(r);count+=len(r)
            vc,vg=measure(data.rows[1],erows[1]);accepted=vc<=limit
            record={'epoch':e+1,'train_ccc':total/count,'train_gene':rna/max(rnacount,1),'validation_ccc':vc,'validation_gene':vg,'CCC_eligible':accepted,
                'auxiliary_conflict_fraction':float(np.mean(conflicts)) if conflicts else 0.,'auxiliary_norm_ratio':float(np.mean(ratios)) if ratios else 0.}
            history.append(record);pd.DataFrame(history).to_csv(out/'joint_training.csv',index=False)
            print(f'Joint epoch {e+1}/{config.joint_epochs}: CCC {total/count:.5f}/{vc:.5f}; gene {rna/max(rnacount,1):.5f}/{vg:.5f}; CCC gate={accepted}',flush=True)
            if accepted and vg<best_gene-1e-6:
                best_gene=vg;best_epoch=e+1;stale=0;best={'gnn_state':copy.deepcopy(model.state_dict()),'gene_state':copy.deepcopy(decoder.state_dict())}
            else:
                stale+=1
                if stale>=config.patience:break
        model.load_state_dict(best['gnn_state']);decoder.load_state_dict(best['gene_state'])
        saved={**best,'best_epoch':best_epoch,'initial_validation_CCC':initial_ccc,'initial_validation_gene':initial_gene,'CCC_validation_limit':limit,'manifest':manifest}
        torch.save(saved,checkpoint)
    hook.remove()
    with torch.no_grad():amplitude.weight.mul_(amplitude_scale);amplitude.bias.mul_(amplitude_scale)
    go=out/'gnn';go.mkdir(exist_ok=True);eo=out/'expression';eo.mkdir(exist_ok=True)
    torch.save({**payload,'model_state':model.state_dict(),'joint_config':config.to_dict(),'joint_best_epoch':best_epoch},go/'checkpoint.pt')
    torch.save({'model_state':decoder.state_dict(),'best_epoch':best_epoch,'config':config.to_dict()},eo/'checkpoint.pt')
    arrays=infer(model,data,go,config.batch_size);z=np.asarray(arrays['Z']);h=np.load(go/'H.npy');np.save(eo/'dictionary.npy',decoder.dictionary.detach().cpu().numpy())
    np.savez(eo/'preprocessing.npz',**{k:pre[k] for k in keys})
    prediction=np.empty_like(target);decoder.eval()
    reconstruction=np.lib.format.open_memmap(go/'CCC_reconstruction.npy',mode='w+',dtype='float32',shape=data.targets.matrix.shape)
    with torch.no_grad():
        for start in range(0,len(z),config.batch_size):
            r=slice(start,start+config.batch_size);prediction[r]=decoder(torch.tensor(z[r],device=device),tensors[2][r]).cpu().numpy();reconstruction[r]=z[r]@h
    reconstruction.flush();del reconstruction
    baseline=np.log1p(norm[:,:,None]*pre['baseline'][None]);np.save(eo/'prediction.npy',prediction);np.save(eo/'baseline_prediction.npy',baseline)
    gene_metrics=metrics(target,prediction,baseline,present,pre['feature_mask'],erows[2],list(enriched.var_names),pd.read_csv(out/'response_gene_design.csv',keep_default_na=False),data.categories,out)
    loading_evaluation(h,np.load(source/'ground_truth/MI_dictionary.npy'),np.load(eo/'dictionary.npy'),list(enriched.var_names),pd.read_csv(out/'response_gene_design.csv',keep_default_na=False),data.categories,out)
    projection=dictionary_program_projection(h,data.feature_scale)
    discovery_cfg=replace(gcfg,output_dir=str(go));k,adaptive,_=discover(data,arrays,h,discovery_cfg,program_projection=projection)
    fixed=fixed_clustering(z@projection,data.rows[0],base,standardize=False)
    pd.DataFrame({'cell_id':adata.obs.cell_id.to_numpy(),'CCC_Phe GNN (fixed K)':fixed,'CCC_Phe GNN (selected K)':adaptive}).to_csv(out/'raw_clusters.csv',index=False)
    # Truth is first read here, after training and unsupervised K selection.
    from .evaluation import label_mapping,mapped,partition_scores
    truth=adata.obs.true_niche.to_numpy();edge=np.isin(truth,[2,3]);records=[]
    old_labels=pd.read_csv(source/'raw_clusters.csv')
    for name,pred in [('CCC-only GNN',old_labels['CCC_Phe GNN (fixed K)'].to_numpy()),('CCC + gene GNN (fixed K)',fixed),('CCC + gene GNN (selected K)',adaptive)]:
        mapping=label_mapping(pred,truth,data.rows[0]);ab=label_mapping(pred,truth,data.rows[0][edge[data.rows[0]]]);aligned=mapped(pred,mapping)
        for scope,r in [('heldout_test',data.rows[2]),('all_transductive',np.arange(len(truth)))]:
            records.append({'method':name,'evaluation':scope,'K':len(np.unique(pred)),**partition_scores(truth,pred,aligned,r,ab,data.rows[0][edge[data.rows[0]]])})
    pd.DataFrame(records).to_csv(out/'metrics.csv',index=False)
    top=[];compositions=[]
    for niche in np.unique(adaptive):
        r=np.flatnonzero(adaptive==niche);mean_z=z[r].mean(0);ccc=mean_z@h
        for rank,f in enumerate(np.argsort(-ccc)[:10],1):
            feature=data.features.iloc[f];p=int(np.argmax(mean_z*h[:,f]));top.append({'niche':int(niche)+1,'rank':rank,'sender':feature.sender,'receiver':feature.receiver,'LR':f'{feature.ligand}–{feature.receptor}','LR_id':feature.lr_id,'mean_reconstructed_CCC':float(ccc[f]),'main_program':p+1})
        compositions.append(np.asarray(data.composition_target[r]).mean(0))
    pd.DataFrame(top).to_csv(out/'niche_top10_CCC.csv',index=False)
    pd.DataFrame(compositions,index=np.unique(adaptive)+1,columns=data.categories).rename_axis('niche').to_csv(out/'niche_cell_composition.csv')
    final_ccc,final_gene=measure(data.rows[1],erows[1]);test_ccc,test_gene=measure(data.rows[2],erows[2])
    control_z=np.load(source/'gnn/Z.npy');delta_z=float(np.linalg.norm(z-control_z)/max(np.linalg.norm(control_z),1e-12))
    summary={'source_CCC_results':str(source),'frozen_control':str(control),'cells':len(adata),'base_genes':adata.n_vars,'expression_genes':enriched.n_vars,'programs':base.programs,'selected_K':int(k),'joint_best_epoch':best_epoch,'joint_model_accepted':best_epoch>0,
        'validation_CCC_before':initial_ccc,'validation_CCC_after':final_ccc,'validation_gene_before':initial_gene,'validation_gene_after':final_gene,'test_CCC':test_ccc,'test_gene':test_gene,'Z_relative_change':delta_z,
        'expression_training':'Joint encoder gradients; asymmetric CCC protection and norm cap; separate H and gene dictionary updates',
        'CCC_validation_tolerance':config.ccc_validation_tolerance,'response_genes_excluded_from_encoder':True,'truth_used_in_training_or_K_selection':False,'clinical_Cox_modified':False}
    for name,digest in snapshots.items():
        if checksum(source/name)!=digest:raise RuntimeError('Original CCC result modified: '+name)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(out/'complete.json').write_text(json.dumps({'manifest':manifest},indent=2)+'\n')
    for value in [data.targets.matrix,*vars(data).values(),*arrays.values()]:
        if isinstance(value,np.memmap):value._mmap.close()
    return out
