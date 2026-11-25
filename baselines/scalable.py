"""All-cell execution adapters for large, multi-sample sections.

GraphST uses algebraically equivalent sparse readout and exact chunked MSE
gradients. STAGATE follows its official spatial-batch tutorial. BANKSY uses
official feature functions, with independent spatial weights per sample.
SpaNiche runs the vendored R/IntNMF objective with sparse Gaussian arithmetic.
"""
from pathlib import Path
import json
import time
import numpy as np
import scipy.sparse as sp
import torch
import pandas as pd
from sklearn.decomposition import PCA
from .adapters import log_expression


def record(output,name,**values):
    (Path(output)/name).write_text(json.dumps(values,indent=2,allow_nan=False)+'\n')


def graphst_backward(model,x,adj,readout,degree,perm,chunk=8192):
    """Same full-graph objective/gradient, without N×N or N×G activations."""
    import torch.nn.functional as F
    transformed=x@model.weight1
    z=torch.spmm(adj,transformed);za=torch.spmm(adj,transformed[perm])
    emb=F.relu(z);emb_a=F.relu(za)
    g=F.normalize(torch.spmm(readout,emb)/degree[:,None],p=2,dim=1).sigmoid()
    ga=F.normalize(torch.spmm(readout,emb_a)/degree[:,None],p=2,dim=1).sigmoid()
    labels=torch.cat((x.new_ones((len(x),1)),x.new_zeros((len(x),1))),1)
    # Native nn.Bilinear backward materializes N x hidden x hidden.
    # The identical two matrix products avoid that temporary on large slides.
    def discriminator(context,positive,negative):
        weight=model.disc.f_k.weight[0];bias=model.disc.f_k.bias
        positive=((positive@weight)*context).sum(1,keepdim=True)
        negative=((negative@weight)*context).sum(1,keepdim=True)
        if bias is not None:positive=positive+bias;negative=negative+bias
        return torch.cat((positive,negative),1)
    contrast=F.binary_cross_entropy_with_logits(discriminator(g,emb,emb_a),labels)
    contrast+=F.binary_cross_entropy_with_logits(discriminator(ga,emb_a,emb),labels)
    # A@(Z@W2) = (A@Z)@W2. Compute exact MSE derivatives in row chunks,
    # then traverse the shared encoder graph once, not once per row chunk.
    pooled=torch.spmm(adj,z)
    with torch.no_grad():
        gradient=torch.empty_like(pooled);gw=torch.zeros_like(model.weight2);sse=0.
        for start in range(0,len(x),chunk):
            stop=min(start+chunk,len(x));error=pooled[start:stop]@model.weight2-x[start:stop]
            sse+=float(error.square().sum());error*=20./x.numel()
            gw+=pooled[start:stop].T@error;gradient[start:stop]=error@model.weight2.T
    contrast.backward(retain_graph=True);pooled.backward(gradient);model.weight2.grad=gw
    return sse/x.numel(),float(contrast.detach())


def graphst_equivalence():
    from .vendor.graphst.model import Encoder_sparse
    import copy
    torch.manual_seed(40700);n,g=37,11
    x=torch.randn(n,g);mask=(torch.rand(n,n)>.8).float();mask.fill_diagonal_(1)
    adj=mask/mask.sum(1,keepdim=True);perm=torch.randperm(n)
    original=Encoder_sparse(g,7,mask);model=copy.deepcopy(original)
    _,rec,ret,reta=original(x,x[perm],adj.to_sparse())
    labels=torch.cat((torch.ones(n,1),torch.zeros(n,1)),1)
    import torch.nn.functional as F
    loss=10*F.mse_loss(rec,x)+F.binary_cross_entropy_with_logits(ret,labels)+F.binary_cross_entropy_with_logits(reta,labels)
    loss.backward();mse,csl=graphst_backward(model,x,adj.to_sparse(),mask.to_sparse(),mask.sum(1),perm,8)
    maximum=0.
    for a,b in zip(original.parameters(),model.parameters()):
        torch.testing.assert_close(a.grad,b.grad,rtol=3e-5,atol=3e-6)
        maximum=max(maximum,float(abs(a.grad-b.grad).max()))
    return dict(native_loss=float(loss.detach()),streamed_loss=10*mse+csl,max_gradient_abs_error=maximum,
                same_graph_all_cells=True,subsampled_training=False)


def graphst(a,graph,cfg,output):
    from .vendor.graphst.model import Encoder_sparse
    from .vendor.graphst.preprocess import preprocess_adj_sparse,fix_seed
    proof=graphst_equivalence();fix_seed(cfg['seed'])
    x=log_expression(a);scale=x.std(0,ddof=1);scale=np.where(scale>1e-8,scale,1.)
    x=np.minimum(x/scale,10.).astype(np.float32)
    g=graph.maximum(graph.T).tocsr();adj=preprocess_adj_sparse(g).to(cfg['device'])
    raw=(g+sp.eye(len(a),format='csr')).tocoo()
    read=torch.sparse_coo_tensor(torch.from_numpy(np.stack((raw.row,raw.col))).long(),
        torch.from_numpy(raw.data.astype(np.float32)),raw.shape,device=cfg['device']).coalesce()
    degree=torch.as_tensor(np.asarray(raw.sum(1)).ravel(),device=cfg['device'],dtype=torch.float32)
    value=torch.as_tensor(x,device=cfg['device']);model=Encoder_sparse(a.n_vars,cfg['graphst_hidden'],read).to(cfg['device'])
    optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=0.)
    history=[];start=0;resume=Path(output)/'resume.pt'
    if resume.exists():
        saved=torch.load(resume,map_location=cfg['device'],weights_only=False)
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);start=saved['epoch'];history=saved['history']
    rng=np.random.default_rng(cfg['seed'])
    # Advance only the row-permutation RNG on resumption.
    for _ in range(start):rng.permutation(len(a))
    for epoch in range(start,cfg['graphst_epochs']):
        tick=time.monotonic();model.train();optimizer.zero_grad(set_to_none=True)
        perm=torch.as_tensor(rng.permutation(len(a)),device=cfg['device'])
        mse,csl=graphst_backward(model,value,adj,read,degree,perm)
        optimizer.step();history.append(dict(epoch=epoch+1,MSE=mse,contrastive=csl,seconds=time.monotonic()-tick))
        if (epoch+1)%10==0 or epoch+1==cfg['graphst_epochs']:
            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch+1,history=history),resume)
            pd.DataFrame(history).to_csv(Path(output)/'training.csv',index=False)
            print('GraphST EPOCH',epoch+1,history[-1],flush=True)
    model.eval()
    with torch.no_grad():
        z=torch.spmm(adj,value@model.weight1);reconstruction= torch.spmm(adj,z)
        # The published reconstructed-expression embedding has rank<=64.
        # Orthonormal coordinates preserve its normalized Euclidean distances
        # exactly, without storing another N×G matrix or changing clustering.
        # This small factorization is equivalent on CPU and avoids allocating
        # a CUDA solver workspace when other authorized jobs share the GPU.
        u,s,vh=torch.linalg.svd(model.weight2.detach().cpu(),full_matrices=False)
        u,s=u.to(cfg['device']),s.to(cfg['device'])
        emb=np.lib.format.open_memmap(Path(output)/'native_embedding.npy',mode='w+',dtype=np.float32,shape=(len(a),len(s)))
        for begin in range(0,len(a),4096):
            block=(reconstruction[begin:begin+4096]@u)*s
            emb[begin:begin+4096]=torch.nn.functional.normalize(block,p=2,dim=1).cpu().numpy()
        emb.flush()
        np.save(Path(output)/'embedding_orthonormal_basis.npy',vh.cpu().numpy())
    torch.save(dict(model_state=model.state_dict(),gene_scale=scale,config=cfg),Path(output)/'checkpoint.pt')
    record(output,'adapter_audit.json',variant='Official GraphST Encoder_sparse; exact full-graph sparse/chunked arithmetic',
        objective='10*MSE + two native contrastive BCE terms',epochs=cfg['graphst_epochs'],all_cells=len(a),
        validation_cells=0,test_cells=0,native_gradient_equivalence=proof,
        embedding='Euclidean isometry of normalized native reconstructed-expression embedding; no per-column scaling')
    return emb


def spatial_batches(a,limit):
    """Quantile spatial subgraphs, following official STAGATE Tutorial8."""
    xy=np.asarray(a.obsm['spatial']);samples=a.obs.sample_id.to_numpy();result=[]
    def split(ids):
        if len(ids)<=limit:result.append(ids);return
        axis=int(np.ptp(xy[ids],axis=0).argmax());order=ids[np.argsort(xy[ids,axis],kind='stable')];half=len(order)//2
        split(order[:half]);split(order[half:])
    for sample in np.unique(samples):split(np.flatnonzero(samples==sample))
    assert np.array_equal(np.sort(np.concatenate(result)),np.arange(len(a)))
    return result


def stagate(a,graph,cfg,output):
    from .vendor.stagate.STAGATE import STAGATE
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed'])
    x=log_expression(a);g=(graph.maximum(graph.T)+sp.eye(len(a),format='csr')).tocsr()
    batches=spatial_batches(a,cfg['stagate_batch_cells']);data=[]
    for ids in batches:
        local=g[ids][:,ids].tocoo();edge=torch.tensor(np.stack((local.row,local.col)),dtype=torch.long)
        data.append(Data(x=torch.from_numpy(x[ids]),edge_index=edge))
    model=STAGATE([a.n_vars]+cfg['stagate_hidden_dims']).to(cfg['device'])
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg['stagate_lr'],weight_decay=cfg['stagate_weight_decay'])
    loader=DataLoader(data,batch_size=1,shuffle=True);history=[];start=0;resume=Path(output)/'resume.pt'
    if resume.exists():
        saved=torch.load(resume,map_location=cfg['device'],weights_only=False)
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);start=saved['epoch'];history=saved['history']
    for epoch in range(start,cfg['stagate_epochs']):
        total=0.;model.train();tick=time.monotonic()
        for batch in loader:
            batch=batch.to(cfg['device']);optimizer.zero_grad(set_to_none=True)
            _,out=model(batch.x,batch.edge_index);loss=torch.nn.functional.mse_loss(out,batch.x)
            if not torch.isfinite(loss):raise ValueError('Nonfinite STAGATE loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();total+=float(loss.detach())*len(batch.x)
        history.append(dict(epoch=epoch+1,MSE=total/len(a),seconds=time.monotonic()-tick,all_cells_seen=len(a)))
        if (epoch+1)%10==0 or epoch+1==cfg['stagate_epochs']:
            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch+1,history=history),resume)
            pd.DataFrame(history).to_csv(Path(output)/'training.csv',index=False)
            print('STAGATE EPOCH',epoch+1,history[-1],flush=True)
    # Exact full-graph encoder inference with two-hop halos and bounded seeds.
    # Decoder outputs are discarded; the full relevant encoder graph is kept.
    embedding=np.empty((len(a),cfg['stagate_hidden_dims'][-1]),np.float32);model.eval()
    with torch.no_grad():
        for seeds in batches:
            local=seeds
            for _ in range(2):local=np.union1d(local,g[local].indices)
            cut=g[local][:,local].tocoo();edge=torch.tensor(np.stack((cut.row,cut.col)),dtype=torch.long,device=cfg['device'])
            z,_=model(torch.as_tensor(x[local],device=cfg['device']),edge)
            embedding[seeds]=z[torch.as_tensor(np.searchsorted(local,seeds),device=cfg['device'])].cpu().numpy()
    torch.save(dict(model_state=model.state_dict(),config=cfg),Path(output)/'checkpoint.pt')
    record(output,'adapter_audit.json',variant='Official STAGATE model and Tutorial8 spatial-batch training',
        tutorial='https://stagate.readthedocs.io/en/latest/T8_Batch.html',all_cells=len(a),
        epochs=cfg['stagate_epochs'],spatial_batches=len(batches),validation_cells=0,test_cells=0,
        encoder_inference='Full graph with exact two-hop encoder halos; no label smoothing')
    return embedding


def banksy(a,cfg,output):
    from .vendor.banksy.main import generate_spatial_weights_fixed_nbrs,concatenate_all
    from .vendor.banksy.embed_banksy import create_nbr_matrix
    full=a.copy();full.X=log_expression(a);xy=a.obsm['spatial'];samples=a.obs.sample_id.to_numpy();weights={}
    for m in (0,1):
        rr=[];cc=[];vv=[]
        for sample in np.unique(samples):
            ids=np.flatnonzero(samples==sample)
            w,_,_=generate_spatial_weights_fixed_nbrs(xy[ids],m=m,
                num_neighbours=min(cfg['banksy_k_geom'],(len(ids)-1)//(m+1)),
                decay_type=cfg['banksy_weight_decay'],verbose=False)
            v=w.tocoo();rr.append(ids[v.row]);cc.append(ids[v.col]);vv.append(v.data)
        weights[m]=sp.csr_matrix((np.concatenate(vv),(np.concatenate(rr),np.concatenate(cc))),shape=(len(a),len(a)))
        sp.save_npz(Path(output)/f'weights_m{m}.npz',weights[m])
    d={cfg['banksy_weight_decay']:{'weights':weights}}
    neighbors=create_nbr_matrix(full,d,cfg['banksy_weight_decay'],1,center=True,variance_balance=False,verbose=False)
    augmented=concatenate_all([full.X,neighbors[0],neighbors[1]],cfg['banksy_lambda'],reference_rows=np.arange(len(a)))
    augmented=np.asarray(augmented,np.float32)
    pca=PCA(n_components=min(cfg['pca_components'],len(a)-1,augmented.shape[1]),svd_solver='randomized',random_state=cfg['seed'])
    embedding=pca.fit_transform(augmented).astype(np.float32)
    import joblib
    joblib.dump(pca,Path(output)/'pca.joblib')
    record(output,'adapter_audit.json',variant='Official BANKSY mean and centered azimuthal Gabor features',
        per_sample_spatial_graph=True,lambda_value=cfg['banksy_lambda'],all_cells=len(a),validation_cells=0,test_cells=0)
    return embedding


def fit(method,a,graph,members,types,categories,cfg,output):
    if method=='GraphST':return graphst(a,graph,cfg,output)
    if method=='STAGATE':return stagate(a,graph,cfg,output)
    if method=='BANKSY':return banksy(a,cfg,output)
    from .spaniche_large import spaniche
    return spaniche(a,members,types,categories,cfg,output)
