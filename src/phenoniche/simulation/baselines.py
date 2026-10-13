"""Same data/splits, distinct information conditions; reuse the actual CCC-vector AE."""
from pathlib import Path
from dataclasses import asdict
from types import SimpleNamespace
import json
import numpy as np,torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from ..data import PreparedInput,resolve_device
from ..config import ExperimentConfig
from ..model import RepresentationAutoencoder
from ..training import (fit_stage1,load_stage1_checkpoint,infer_latent,initialize_program_decoder,fit_stage2a,fit_stage2b,load_stage2_checkpoint,infer_programs)
from ..gnn.model import physical_weights

def fixed_clustering(values,train,config):
    scaler=StandardScaler().fit(values[train]);x=scaler.transform(values)
    model=MiniBatchKMeans(n_clusters=config.fixed_k,random_state=config.seed,n_init=10,batch_size=min(1024,len(train))).fit(x[train])
    return model.predict(x).astype(np.int16)

def expression_pca(data,config):
    # Pool raw expression by pair degree, without any cell-type labels.
    raw=data.expression.toarray();normalized=np.log1p(raw*10000/np.maximum(raw.sum(1,keepdims=True),1))
    mean=np.empty_like(normalized)
    for start in range(0,len(raw),512):
        stop=min(start+512,len(raw));local=data.members[start:stop]
        weight=physical_weights(torch.from_numpy(data.coordinates[local]),torch.from_numpy(np.asarray(data.weights[start:stop]).copy()),20.,torch.from_numpy(np.asarray(local))).sum(2).numpy()
        mean[start:stop]=(normalized[local]*weight[...,None]).sum(1)/np.maximum(weight.sum(1)[:,None],1e-8)
    scaler=StandardScaler().fit(mean[data.rows[0]])
    pca=PCA(n_components=min(30,len(data.genes),len(data.rows[0])-1),random_state=config.seed).fit(scaler.transform(mean[data.rows[0]]))
    return pca.transform(scaler.transform(mean)).astype(np.float32)

def vector_ae(data,config,output):
    output=Path(output);output.mkdir(exist_ok=True,parents=True)
    if all((output/n).is_file() for n in ('Z.npy','P.npy','H.npy','activity.npy','complete.json')):
        return {name:np.load(output/(name+'.npy'),mmap_mode='r') for name in ('Z','P','H','activity')}
    columns=np.flatnonzero(data.feature_mask);features=data.features.iloc[columns].reset_index(drop=True)
    prepared=PreparedInput(Path(data.targets.matrix.filename),data.targets.matrix.shape,columns,data.feature_scale[columns],data.rows[0],data.rows[1],features,data.cells,data.coordinates,output)
    # Reuse the archived recipe verbatim; adapt its scalar configuration to
    # the same simulation seed/budgets without changing the model architecture.
    values=asdict(ExperimentConfig());values.update(seed=config.seed,device=config.device,batch_size=config.batch_size,stage1_max_epochs=config.ae_stage1_epochs,stage2a_max_epochs=config.ae_stage2_epochs,stage2b_max_epochs=config.ae_stage2_epochs,stage1_patience=config.patience,stage2a_patience=config.patience,stage2b_patience=config.patience,stage2a_head_warmup_epochs=min(2,config.ae_stage2_epochs-1))
    cfg=SimpleNamespace(**values);device=resolve_device(cfg.device)
    p1=output/'stage1_best_checkpoint.pt'
    if p1.exists():
        model=RepresentationAutoencoder(len(columns),32,cfg.encoder_hidden_1,cfg.encoder_hidden_2,cfg.dropout).to(device);load_stage1_checkpoint(p1,model,device)
    else:model,_=fit_stage1(prepared,cfg,output)
    refpath=output/'Z0.npy';reference=np.load(refpath,mmap_mode='r') if refpath.exists() else infer_latent(model,prepared,cfg,refpath)
    program=initialize_program_decoder(prepared,cfg,device)
    p2a=output/'stage2a_best_checkpoint.pt';p2b=output/'stage2b_best_checkpoint.pt'
    if p2b.exists():load_stage2_checkpoint(p2b,model,program,device)
    else:
        if p2a.exists():program.load_state_dict(torch.load(p2a,map_location=device,weights_only=False)['program_state'])
        else:fit_stage2a(model,program,prepared,cfg,output)
        fit_stage2b(model,program,prepared,reference,cfg,output)
    arrays=infer_programs(model,program,prepared,cfg,{k:output/(k+'.npy') for k in ('E32','P32','Z32','Q8_direct','activity')})
    native=program.dictionary.detach().cpu().numpy()*data.feature_scale[columns];mass=native.sum(1);native/=np.maximum(mass[:,None],1e-8)
    h=np.zeros((32,len(data.features)),np.float32);h[:,columns]=native
    z=np.asarray(arrays['Z32'])*mass;np.save(output/'Z.npy',z);np.save(output/'P.npy',np.asarray(arrays['P32']));np.save(output/'H.npy',h);np.save(output/'activity.npy',z.sum(1))
    (output/'complete.json').write_text(json.dumps({'model':'original CCC-vector dual-decoder AE','information':'precomputed CCC input','simulation_signature':config.signature(),'same_spatial_folds':True},indent=2))
    return {'Z':z,'P':arrays['P32'],'H':h,'activity':z.sum(1)}

def run_baselines(data,config,output):
    output=Path(output);output.mkdir(exist_ok=True,parents=True)
    values=expression_pca(data,config)
    ae=vector_ae(data,config,output/'ccc_ae')
    labels={'Cell Composition':fixed_clustering(np.asarray(data.composition_target),data.rows[0],config),
            'Expression PCA':fixed_clustering(values,data.rows[0],config),
            'CCC-AE':fixed_clustering(np.column_stack((np.sqrt(np.asarray(ae['P'])),.25*np.log1p(ae['activity']))),data.rows[0],config)}
    np.save(output/'expression_PCA.npy',values)
    return labels,ae
