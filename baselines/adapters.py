"""Audited upstream APIs; inputs never contain labels, CCC or appended genes."""
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import anndata as ad
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def clean_input(expression,genes,coordinates,samples):
    a=ad.AnnData(sp.csr_matrix(expression,dtype=np.float32))
    a.var_names=list(genes);a.obsm['spatial']=np.asarray(coordinates,np.float32)
    a.obs['sample_id']=np.asarray(samples).astype(str)
    return a


def graph_from_members(members):
    n=len(members);row=np.repeat(np.arange(n),members.shape[1]);col=members.ravel()
    keep=row!=col
    return sp.csr_matrix((np.ones(keep.sum(),np.float32),(row[keep],col[keep])),shape=(n,n))


def log_expression(a):
    raw=a.X.toarray() if sp.issparse(a.X) else np.asarray(a.X)
    return np.log1p(raw*10000/np.maximum(raw.sum(1,keepdims=True),1)).astype(np.float32)


def graphst(a,graph,train_cells,config,output):
    from .vendor.graphst import GraphST
    from .vendor.graphst.model import Encoder_sparse
    from .vendor.graphst.preprocess import preprocess_adj_sparse
    x=log_expression(a)
    # Same published scale convention (zero_center=False, max_value=10),
    # but moments fitted ONLY on cells used by the original GNN training fold.
    scale=np.std(x[train_cells],axis=0,ddof=1);scale=np.where(scale>1e-8,scale,1.)
    x=np.minimum(x/scale,10.).astype(np.float32)
    symmetric=graph.maximum(graph.T);local=symmetric[train_cells][:,train_cells].toarray().astype(np.float32)
    tr=a[train_cells].copy();tr.X=x[train_cells];tr.var['highly_variable']=True
    tr.obsm['adj']=local;tr.obsm['graph_neigh']=local
    device=torch.device(config['device'])
    runner=GraphST(tr,device=device,epochs=config['graphst_epochs'],dim_output=config['graphst_hidden'],random_seed=config['seed'],datatype='Slide')
    result=runner.train();training=np.asarray(result.obsm['emb'],np.float32)
    # Published node-independent weights support a new graph at inference.
    # No held-out expression enters optimization; graph-bound readout is
    # reinstantiated at the new node count, then receives the learned weights.
    neighbor=torch.from_numpy(symmetric.toarray().astype(np.float32)+np.eye(len(a),dtype=np.float32)).to(device)
    model=Encoder_sparse(x.shape[1],config['graphst_hidden'],neighbor).to(device)
    model.load_state_dict(runner.model.state_dict());model.eval()
    xx=torch.from_numpy(x).to(device);adjacency=preprocess_adj_sparse(symmetric).to(device)
    with torch.no_grad():embedding=torch.nn.functional.normalize(model(xx,xx,adjacency)[1],p=2,dim=1).cpu().numpy()
    torch.save({'model_state':runner.model.state_dict(),'training_cells':train_cells,'gene_scale':scale,'config':config,'genes':list(a.var_names)},Path(output)/'checkpoint.pt')
    np.save(Path(output)/'training_embedding.npy',training)
    return embedding,training


def cellcharter(a,graph,train_cells,train_centers,config,output):
    import cellcharter as cc
    # CellCharter accepts arbitrary expression representations. This is
    # explicitly the PCA variant, NOT claimed to be its scVI tutorial recipe.
    if config['cellcharter_representation']!='PCA':raise ValueError('Audited adapter implements CellCharter(PCA); do not silently substitute scVI')
    x=log_expression(a);scaler=StandardScaler().fit(x[train_cells]);scaled=scaler.transform(x)
    pca=PCA(n_components=min(config['pca_components'],len(train_cells)-1,x.shape[1]),random_state=config['seed']).fit(scaled[train_cells])
    full=a.copy();full.obsm['X_pca']=pca.transform(scaled).astype(np.float32);full.obsp['shared_knn_connectivities']=graph
    cc.gr.aggregate_neighbors(full,n_layers=config['cellcharter_layers'],use_rep='X_pca',connectivity_key='shared_knn',sample_key='sample_id')
    tr=full[train_cells].copy();tr.obsp['shared_knn_connectivities']=graph[train_cells][:,train_cells]
    cc.gr.aggregate_neighbors(tr,n_layers=config['cellcharter_layers'],use_rep='X_pca',connectivity_key='shared_knn',sample_key='sample_id')
    positions=np.searchsorted(train_cells,train_centers)
    if not np.array_equal(train_cells[positions],train_centers):raise ValueError('Train-center/cell alignment error')
    fitting=tr[positions].copy()
    if config['device'].startswith('cuda:'):
        trainer={'accelerator':'gpu','devices':[int(config['device'].split(':')[1])]}
    else:trainer={'accelerator':'cpu','devices':1}
    trainer.update(max_epochs=config['gmm_epochs'],logger=False,enable_progress_bar=False)
    clustering=cc.tl.Cluster(n_clusters=config['fixed_k'],random_state=config['seed'],trainer_params=trainer)
    clustering.fit(fitting,use_rep='X_cellcharter')
    # CellCharter.predict invokes a Lightning DataLoader wrapper incompatible
    # with the installed Lightning 1.9/Torch 2.3 combination. Use the public
    # fitted TorchGMM module's exact log-responsibility/argmax prediction instead;
    # the EM fit, means, covariances and posterior calculation remain upstream.
    mixture=clustering.model_.to(torch.device(config['device']));mixture.eval();predictions=[]
    with torch.no_grad():
        for start in range(0,len(full),2048):
            value=torch.as_tensor(full.obsm['X_cellcharter'][start:start+2048],device=config['device'])
            posterior,_=mixture(value);predictions.append(posterior.argmax(-1).cpu().numpy())
    labels=np.concatenate(predictions).astype(int)
    clustering.save(Path(output)/'gmm')
    import joblib
    joblib.dump({'scaler':scaler,'pca':pca},Path(output)/'expression_preprocessing.joblib')
    return np.asarray(full.obsm['X_cellcharter'],np.float32),np.asarray(fitting.obsm['X_cellcharter'],np.float32),labels
