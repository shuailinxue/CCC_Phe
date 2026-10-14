"""Sampled, read-only diagnosis of the gene-rich simulation on GPU 2.

Truth-dependent classifiers/MI summaries are explicitly evaluation diagnostics,
never training targets or checkpoint-selection criteria for a GNN.
"""
from pathlib import Path
from dataclasses import replace
from functools import partial
import argparse,copy,json,sys,time
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score,accuracy_score,adjusted_rand_score
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from phenoniche.storage import STORAGE_ROOT
from phenoniche.gnn.data import GraphData,Targets
from phenoniche.gnn.config import GNNConfig
from phenoniche.gnn.model import model_from_config,physical_weights
from phenoniche.gnn.training import initialize,epoch,make_optimizer
from phenoniche.gnn.losses import ccc_loss
from phenoniche.simulation.baselines import dictionary_program_projection
from phenoniche.simulation.evaluation import label_mapping,mapped,partition_scores
from phenoniche.simulation.response_simulation import checksum


def read_data(source):
    payload=torch.load(source/'gnn/checkpoint.pt',map_location='cpu',weights_only=False)
    a=ad.read_h5ad(source/'injected.h5ad');cache=source/'input';pre=np.load(cache/'preprocessing.npz')
    categories=tuple(payload['categories']);types=pd.Categorical(a.obs.cell_type,categories=categories).codes
    folds=payload['folds'];data=GraphData(sp.csr_matrix(a.X),np.asarray(a.X).sum(1),tuple(a.var_names),payload['encoder_indices'],payload['gene_mean'],payload['gene_scale'],types,categories,np.asarray(a.obsm['spatial']),np.load(cache/'members.npy',mmap_mode='r'),np.load(cache/'anchor_weights.npy',mmap_mode='r'),Targets(np.load(cache/'ccc.npy',mmap_mode='r'),np.arange(len(payload['feature_mask']))),payload['feature_scale'],payload['feature_mask'],pd.DataFrame(payload['features']),a.obs[['cell_id','cell_type']].reset_index(drop=True),[np.flatnonzero(folds==k) for k in range(3)],folds,pre['blocks'],np.load(cache/'composition_target.npy',mmap_mode='r'),payload['data_audit'])
    data.feature_mean=payload['feature_mean'];data.feature_nonzero_fraction=payload['feature_nonzero_fraction']
    return a,data,payload


def diagnostics(a,data,source,out):
    train,val,test=data.rows;truth=a.obs.true_niche.to_numpy();edge=np.isin(truth,[2,3]);etr=train[edge[train]];eva=val[edge[val]];ete=test[edge[test]]
    true_h=np.load(source/'ground_truth/MI_dictionary.npy');c=np.asarray(data.targets.matrix)
    score=c@true_h.T/np.maximum(true_h.sum(1),1e-12);np.save(out/'observed_MI_CCC_means.npy',score)
    records=[]
    for feature_name,x in [('Known MI1/MI2 CCC (target-informed diagnostic)',score[:,:2]),('All observed CCC',c[:,data.feature_mask]/data.feature_scale[data.feature_mask])]:
        clf=make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=2000,random_state=40700)).fit(x[etr],(truth[etr]==3).astype(int))
        for name,r in [('validation',eva),('test',ete)]:
            probability=clf.predict_proba(x[r])[:,1];records.append({'representation':feature_name,'scope':name,'edge_n':len(r),'AB_AUC':roc_auc_score(truth[r]==3,probability),'AB_accuracy':accuracy_score(truth[r]==3,probability>.5),'supervised_diagnostic_not_benchmark':True})
    for name,path in [('CCC-only GNN',source/'gnn'),('Joint GNN',Path(str(source).replace('gene_rich_simulation','gene_rich_expression_joint'))/'gnn')]:
        z=np.load(path/'Z.npy');h=np.load(path/'H.npy');prediction=z@(h@true_h.T)/np.maximum(true_h.sum(1),1e-12)
        np.save(out/(name.replace(' ','_')+'_MI_CCC_means.npy'),prediction)
        for feature_name,x in [(name+' Z',np.log1p(z)),(name+' reconstructed MI CCC',prediction[:,:2])]:
            clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,random_state=40700)).fit(x[etr],truth[etr]==3)
            for scope,r in [('validation',eva),('test',ete)]:
                probability=clf.predict_proba(x[r])[:,1];records.append({'representation':feature_name,'scope':scope,'edge_n':len(r),'AB_AUC':roc_auc_score(truth[r]==3,probability),'AB_accuracy':accuracy_score(truth[r]==3,probability>.5),'supervised_diagnostic_not_benchmark':True})
    pd.DataFrame(records).to_csv(out/'signal_separability.csv',index=False)
    pd.DataFrame(score,columns=['MI1','MI2','MI3','MI4']).assign(region=truth).groupby('region').agg(['mean','std']).to_csv(out/'region_observed_CCC.csv')
    # Unsupervised clustering of the actually observed supervision, without truth.
    frames=[]
    rng=np.random.default_rng(40700);fit=np.sort(rng.choice(train,min(2500,len(train)),replace=False))
    for name,x in [('Observed CCC native scaled',c[:,data.feature_mask]/data.feature_scale[data.feature_mask]),('Observed CCC log scaled',np.log1p(c[:,data.feature_mask]/data.feature_scale[data.feature_mask]))]:
        labels=MiniBatchKMeans(n_clusters=4,random_state=40700,n_init=10,batch_size=1024).fit(x[fit]).predict(x)
        mapping=label_mapping(labels,truth,train);ab=label_mapping(labels,truth,etr)
        for scope,r in [('all',np.arange(len(truth))),('test',test)]:frames.append({'representation':name,'scope':scope,**partition_scores(truth,labels,mapped(labels,mapping),r,ab,etr)})
    pd.DataFrame(frames).to_csv(out/'observed_CCC_clustering.csv',index=False)
    lr=pd.read_csv(source/'lr.csv');before=ad.read_h5ad(source/'before_injection.h5ad');roles=[]
    for mi in range(1,5):
        links=lr[lr.mi==mi]
        for side,role in [('ligand',links.sender_role.iloc[0]),('receptor',links.receiver_role.iloc[0])]:
            g=a.var_names.get_indexer(links[side]);r=np.flatnonzero((a.obs.cell_type==role)&(np.isin(truth,[2,3]) if mi<3 else truth==(0 if mi==3 else 1)))
            b=np.asarray(before.X[r])[:,g];v=np.asarray(a.X[r])[:,g]
            roles.append({'MI':mi,'side':side,'type':role,'cells':len(r),'before_mean':float(b.mean()),'after_mean':float(v.mean()),'before_zero_fraction':float((b==0).mean()),'after_zero_fraction':float((v==0).mean()),'before_CV':float(b.std()/max(b.mean(),1e-12)),'after_CV':float(v.std()/max(v.mean(),1e-12))})
    pd.DataFrame(roles).to_csv(out/'LR_dropout_noise.csv',index=False)
    print(pd.DataFrame(records).to_string(index=False),flush=True)


def evaluate_prediction(a,data,rows,z,h,train,out,name):
    projection=dictionary_program_projection(h,data.feature_scale);x=z@projection
    model=MiniBatchKMeans(n_clusters=4,random_state=40700,n_init=10,batch_size=min(1024,len(train))).fit(x[:len(train)])
    labels=model.predict(x);truth=a.obs.true_niche.to_numpy()[rows]
    tr=np.arange(len(train));edge=np.isin(truth,[2,3]);edge_train=tr[edge[tr]]
    mapping=label_mapping(labels,truth,tr);ab=label_mapping(labels,truth,edge_train);aligned=mapped(labels,mapping)
    records=[];offset=len(train)
    for scope,r in [('train',tr),('validation',np.arange(offset,offset+len(data.rows[1]))),('test',np.arange(offset+len(data.rows[1]),len(rows)))]:
        records.append({'experiment':name,'scope':scope,**partition_scores(truth,labels,aligned,r,ab,edge_train)})
    np.save(out/'sample_Z.npy',z);np.save(out/'H.npy',h);np.save(out/'raw_labels.npy',labels)
    return records


def sampled_fit(a,original_data,payload,out,name,train_n=1600,epochs=25):
    data=copy.copy(original_data);rng=np.random.default_rng(40700)
    original_rows=original_data.rows
    data.rows=[np.sort(rng.choice(r,min(len(r),train_n if k==0 else 600),replace=False)) for k,r in enumerate(original_rows)]
    cfg=replace(GNNConfig(**payload['config']),device='cuda:2',max_epochs=epochs,patience=epochs,output_dir=str(out),batch_size=128,nmf_initialization_sample_size=max(32,len(data.rows[0])))
    if name.startswith('balanced'):
        # Equal training-RMS units: preserve CCC values, fit scale on train only.
        target=data.targets[data.rows[0]]
        data.feature_scale=np.maximum(np.sqrt(np.mean(target.astype(np.float64)**2,axis=0)),.05).astype(np.float32)
        cfg=replace(cfg,dictionary_parameterization='projected',dictionary_coordinates='feature_scaled',dictionary_lr=.0005)
    if 'light_reg' in name:cfg=replace(cfg,lambda_reg=.0005)
    if name.startswith('balanced') and 'pair' in name:cfg=replace(cfg,program_pair_pooling=True,lr_pair_pooling=True)
    if name=='deeper':cfg=replace(cfg,hidden=128,encoder_depth=2,message_layers=2,dropout=.1,architecture='expression_gnn_residual_v2',lr=.0005)
    if name=='lr_pair':cfg=replace(cfg,program_pair_pooling=True,lr_pair_pooling=True,lr=.0005)
    if name=='projected':cfg=replace(cfg,dictionary_parameterization='projected',dictionary_coordinates='feature_scaled',dictionary_lr=.0002)
    if name=='lr_smooth_warm':cfg=replace(cfg,program_pair_pooling=True,lr_pair_pooling=True,lr=.0005)
    if name=='smooth60_pair_warm':cfg=replace(cfg,program_pair_pooling=True,lr_pair_pooling=True,lr=.0005)
    if name=='lr_markers':
        keep=set(data.features.ligand)|set(data.features.receptor)|{g for g in data.genes if g.startswith('SYN_MARKER_')}
        select=np.asarray([i for i,p in enumerate(data.encoder_indices) if data.genes[p] in keep]);data.encoder_indices=data.encoder_indices[select];data.gene_mean=data.gene_mean[select];data.gene_scale=data.gene_scale[select]
        cfg=replace(cfg,encoder_genes=len(select))
    init_data=data;teacher_target=data.targets[data.rows[0]]
    if name=='opportunity_impute':
        # Initialization diagnostic only: fill structurally unobserved type
        # pairs using TRAINING supported means, never validation/test targets.
        send=np.asarray([data.categories.index(t) for t in data.features.sender]);receive=np.asarray([data.categories.index(t) for t in data.features.receiver]);observed=[]
        for start in range(0,len(data.rows[0]),128):
            rows=data.rows[0][start:start+128];members=np.asarray(data.members[rows]);local_type=torch.tensor(data.types[members],device='cuda:2',dtype=torch.long)
            indicator=torch.nn.functional.one_hot(local_type,len(data.categories)).float()
            pair=physical_weights(torch.tensor(data.coordinates[members],device='cuda:2'),torch.tensor(np.asarray(data.weights[rows]).copy(),device='cuda:2'),cfg.sigma_um,torch.tensor(members,device='cuda:2'))
            opportunities=indicator.transpose(1,2)@pair@indicator
            observed.append((opportunities[:,send,receive]>1e-4).cpu().numpy())
        observed=np.concatenate(observed);mean=(teacher_target*observed).sum(0)/np.maximum(observed.sum(0),1)
        teacher_target=np.where(observed,teacher_target,mean).astype(np.float32)
        class ObservedTargets:
            def __getitem__(self,rows):return teacher_target[np.searchsorted(data.rows[0],rows)]
        init_data=copy.copy(data);init_data.targets=ObservedTargets()
    if 'smooth' in name:
        from scipy.spatial import cKDTree
        count,sigma=(64,60.) if 'smooth60' in name else (16,20.)
        distance,neighbors=cKDTree(data.coordinates[data.rows[0]]).query(data.coordinates[data.rows[0]],k=count)
        weights=np.exp(-distance**2/(2*sigma**2));weights/=weights.sum(1,keepdims=True)
        matrix=sp.csr_matrix((weights.ravel(),(np.repeat(np.arange(len(neighbors)),count),neighbors.ravel())),shape=(len(neighbors),len(neighbors)))
        teacher_target=(matrix@teacher_target).astype(np.float32)
        class InitialTargets:
            def __getitem__(self,rows):return teacher_target[np.searchsorted(data.rows[0],rows)]
        init_data=copy.copy(data);init_data.targets=InitialTargets()
    model=initialize(init_data,cfg).to('cuda:2');amplitude=model.activity_head;factor=max(float(amplitude.bias.detach().abs().max()),1.)
    if name.startswith('lr_skip'):
        # A diagnostic direct expression/spatial route. No observed cell type,
        # precomputed CCC, region or true-MI membership enters this branch.
        catalog=data.features[['lr_id','ligand','receptor']].drop_duplicates()
        gene_indices=sorted(set(catalog.ligand)|set(catalog.receptor))
        data.enable_lr_inputs(np.asarray([data.genes.index(g) for g in gene_indices]))
        position={g:i for i,g in enumerate(gene_indices)}
        ligand=torch.tensor([position[g] for g in catalog.ligand],device='cuda:2')
        receptor=torch.tensor([position[g] for g in catalog.receptor],device='cuda:2')
        model.lr_skip_head=torch.nn.Linear(len(catalog),cfg.programs,bias=False).to('cuda:2')
        torch.nn.init.zeros_(model.lr_skip_head.weight)
        original_forward=model.forward
        def skip_forward(expression,local_index,coordinates,anchor_weights,cell_ids=None,**kwargs):
            raw=kwargs.pop('raw_lr_expression')
            pair=physical_weights(coordinates,anchor_weights,cfg.sigma_um,cell_ids)
            s=raw[:,ligand].clamp_min(0).sqrt()[local_index]
            r=raw[:,receptor].clamp_min(0).sqrt()[local_index]
            signal=(s*(pair@r)).sum(1)/pair.sum((1,2)).clamp_min(1e-12)[:,None]
            model._skip_signal=torch.log1p(signal)
            return original_forward(expression,local_index,coordinates,anchor_weights,cell_ids,**kwargs)
        model.forward=skip_forward
        model.composition_head.register_forward_hook(lambda module,args,value:value+model.lr_skip_head(model._skip_signal))
        # GraphData supplies raw expressions only when this flag is set.
        model.lr_pair_pooling=True
    if model.lr_pair_pooling and data.raw_lr_indices is None:data.enable_lr_inputs(init_data.raw_lr_indices)
    with torch.no_grad():amplitude.weight.div_(factor);amplitude.bias.div_(factor)
    hook=amplitude.register_forward_hook(lambda module,args,value:value*factor)
    allrows=np.concatenate(data.rows);data.prepare_subset_device(allrows,'cuda:2')
    optimizer=make_optimizer(model,cfg)
    objective=partial(ccc_loss,kind='huber',delta=1. if name.startswith('balanced') else 3.)
    if name.startswith('opportunity'):
        types=torch.as_tensor(data.types[data._resident_cell_ids],device='cuda:2',dtype=torch.long)
        sender=torch.tensor([data.categories.index(t) for t in data.features.sender],device='cuda:2')
        receiver=torch.tensor([data.categories.index(t) for t in data.features.receiver],device='cuda:2')
        def opportunity_loss(prediction,target,scale,mask,batch):
            with torch.no_grad():
                local_types=types[batch['cell_ids']]
                indicator=torch.nn.functional.one_hot(local_types,len(data.categories)).float()
                pair=physical_weights(batch['coordinates'],batch['anchor_weights'],cfg.sigma_um,batch['cell_ids'])
                opportunities=indicator.transpose(1,2)@pair@indicator
                weight=(opportunities[:,sender,receiver]>1e-4).float()
                weight=weight/weight[:,mask].mean().clamp_min(1e-8)
            return ccc_loss(prediction,target,scale,mask,kind='huber',delta=3.,entry_weights=weight)
        opportunity_loss.uses_batch=True
        objective=opportunity_loss
    best=float('inf');state=None;history=[];started=time.monotonic()
    if 'warm' in name:
        # Train-only NMF activity distillation. Validation/test CCC never fits
        # teacher codes. The main supervised CCC values remain untouched.
        mask=data.feature_mask;scale=torch.as_tensor(data.feature_scale[mask],device='cuda:2')
        dictionary=model.dictionary.detach()[:,mask]/scale
        target=torch.as_tensor(teacher_target[:,mask],device='cuda:2')/scale
        with torch.no_grad():
            codes=torch.full((len(target),cfg.programs),factor/cfg.programs,device='cuda:2')
            numerator=target@dictionary.T;gram=dictionary@dictionary.T
            for step in range(300):codes*=numerator/(codes@gram).clamp_min(1e-10)
            log_codes=torch.log1p(codes);mean=log_codes.mean(0);std=log_codes.std(0).clamp_min(.03 if name.startswith('balanced') or 'smooth60' in name else .3)
        np.save(out/'training_teacher_Z.npy',codes.cpu().numpy())
        proportional=name=='proportion_warm'
        if proportional:
            codes=codes/codes.sum(1,keepdim=True).clamp_min(1e-8)
            root_codes=codes.sqrt();mean=root_codes.mean(0);std=root_codes.std(0).clamp_min(.03)
            with torch.no_grad():
                vtarget=torch.tensor(data.targets[data.rows[1]][:,mask],device='cuda:2')/scale
                vcode=torch.full((len(vtarget),cfg.programs),1.,device='cuda:2');vn=vtarget@dictionary.T
                for step in range(300):vcode*=vn/(vcode@gram).clamp_min(1e-10)
                vcode=(vcode/vcode.sum(1,keepdim=True).clamp_min(1e-8)).sqrt()
        lookup=np.full(len(data.cells),-1,np.int64);lookup[data.rows[0]]=np.arange(len(data.rows[0]))
        encoder=[p for n,p in model.named_parameters() if n not in ('raw_dictionary','raw_composition') and not (proportional and n.startswith('activity_head'))]
        warm_optimizer=torch.optim.AdamW(encoder,lr=.0005,weight_decay=.0001);warm_history=[]
        for e in range(25):
            model.train();rows=np.random.default_rng(40700+e).permutation(data.rows[0]);losses=[]
            for start in range(0,len(rows),128):
                r=rows[start:start+128];result=data.forward(model,data.batch(r,'cuda:2'),reconstruct=False)
                prediction=((result['P'].clamp_min(1e-12).sqrt() if proportional else torch.log1p(result['Z']))-mean)/std;truth_code=((root_codes if proportional else log_codes)[lookup[r]]-mean)/std
                loss=torch.nn.functional.huber_loss(prediction,truth_code)
                warm_optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(encoder,5.);warm_optimizer.step();losses.append(float(loss.detach()))
            va=epoch(model,data,data.rows[1],cfg,None,e,objective);score=va['ccc']+cfg.lambda_reg*va['regularization']
            if proportional:
                model.eval();vprediction=[]
                with torch.no_grad():
                    for start in range(0,len(data.rows[1]),128):vprediction.append(data.forward(model,data.batch(data.rows[1][start:start+128],'cuda:2'),reconstruct=False)['P'])
                    vp=torch.cat(vprediction).clamp_min(1e-12).sqrt()
                    score=float(torch.nn.functional.huber_loss((vp-mean)/std,(vcode-mean)/std))
            warm_history.append({'epoch':e+1,'teacher_loss':float(np.mean(losses)),'validation_CCC':va['ccc']})
            if score<best:best=score;state=copy.deepcopy(model.state_dict());best_epoch=-(e+1)
            if e%5==0:print(name,'warm',e+1,'teacher',round(float(np.mean(losses)),5),'validation',round(va['ccc'],5),flush=True)
        pd.DataFrame(warm_history).to_csv(out/'warm_training.csv',index=False)
        model.load_state_dict(state)
        if proportional:
            model.eval();warm_z=[]
            with torch.no_grad():
                for start in range(0,len(allrows),128):warm_z.append(data.forward(model,data.batch(allrows[start:start+128],'cuda:2'),reconstruct=False)['Z'].cpu().numpy())
            warm_out=out/'warm_stage';warm_out.mkdir(exist_ok=True)
            warm_records=evaluate_prediction(a,data,allrows,np.concatenate(warm_z),model.dictionary.detach().cpu().numpy(),data.rows[0],warm_out,name)
            pd.DataFrame(warm_records).to_csv(warm_out/'metrics.csv',index=False)
            best=float('inf')
    for e in range(epochs):
        tr=epoch(model,data,data.rows[0],cfg,optimizer,e,objective);va=epoch(model,data,data.rows[1],cfg,None,e,objective)
        score=va['ccc']+cfg.lambda_reg*va['regularization'];history.append({'epoch':e+1,**{'train_'+k:v for k,v in tr.items()},**{'validation_'+k:v for k,v in va.items()}})
        pd.DataFrame(history).to_csv(out/'training.csv',index=False)
        if score<best:best=score;state=copy.deepcopy(model.state_dict());best_epoch=e+1
        if e%5==0:print(name,e+1,'train',round(tr['ccc'],5),'validation',round(va['ccc'],5),flush=True)
    model.load_state_dict(state);model.eval();values=[]
    with torch.no_grad():
        for start in range(0,len(allrows),128):values.append(data.forward(model,data.batch(allrows[start:start+128],'cuda:2'))['Z'].cpu().numpy())
    z=np.concatenate(values);h=model.dictionary.detach().cpu().numpy()
    metrics=evaluate_prediction(a,data,allrows,z,h,data.rows[0],out,name)
    hook.remove()
    with torch.no_grad():amplitude.weight.mul_(factor);amplitude.bias.mul_(factor)
    torch.save({'model_state':model.state_dict(),'config':cfg.to_dict(),'architecture':cfg.architecture,'activity_parameter_scale':factor,'genes':data.genes,'categories':data.categories,'features':data.features.to_dict('list'),'encoder_indices':data.encoder_indices,'gene_mean':data.gene_mean,'gene_scale':data.gene_scale,'feature_mask':data.feature_mask,'feature_scale':data.feature_scale,'feature_mean':data.feature_mean,'feature_nonzero_fraction':data.feature_nonzero_fraction,'folds':data.folds,'data_audit':data.audit,'expression_normalization':data.expression_normalization,'library_mean':data.library_mean,'library_scale':data.library_scale,'raw_lr_indices':data.raw_lr_indices,'best_epoch':best_epoch,'sample_rows':data.rows,'source_checkpoint_sha256':checksum(Path(payload['config']['output_dir'])/'checkpoint.pt'),'diagnostic_only_forward':name.startswith('lr_skip')},out/'checkpoint.pt')
    np.savez(out/'sample_rows.npz',train=data.rows[0],validation=data.rows[1],test=data.rows[2])
    pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False)
    (out/'summary.json').write_text(json.dumps({'best_epoch':best_epoch,'validation_selection_loss':best,'input_genes':len(data.encoder_indices),'runtime_seconds':time.monotonic()-started,'truth_used_in_training_or_checkpoint_selection':False},indent=2))
    del data._resident;torch.cuda.empty_cache()
    return metrics


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['audit','fit'],default='audit');parser.add_argument('--experiments',default='standard,lr_markers,deeper,lr_pair');parser.add_argument('--train-n',type=int,default=1600);parser.add_argument('--epochs',type=int,default=25);args=parser.parse_args()
    torch.set_num_threads(6);torch.cuda.set_device(2)
    source=STORAGE_ROOT/'outputs/tumor_edge_gene_rich_simulation';out=STORAGE_ROOT/'outputs/tumor_edge_gene_rich_diagnostics';out.mkdir(exist_ok=True)
    a,data,payload=read_data(source)
    if args.phase=='audit':diagnostics(a,data,source,out)
    else:
        records=[]
        for name in args.experiments.split(','):
            directory=out/name;directory.mkdir(exist_ok=True)
            if (directory/'summary.json').exists():records+=pd.read_csv(directory/'metrics.csv').to_dict('records');continue
            records+=sampled_fit(a,data,payload,directory,name,args.train_n,args.epochs)
            pd.DataFrame(records).to_csv(out/'sampled_experiment_metrics.csv',index=False)

if __name__=='__main__':main()
