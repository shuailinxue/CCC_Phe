"""Full-cell KMeans on the trained, globally calibrated compact h."""
import numpy as np
from pathlib import Path
import json
from .io import output_path, bind_backend, array_identity, write_json, digest


def gpu_kmeans(embedding,n_clusters,seed=40700,n_init=10,device='cuda:2',report=None,max_iter=300):
    """Full-batch Lloyd KMeans, greedy kmeans++ initialization, no cell sampling."""
    import torch
    from types import SimpleNamespace
    x=torch.as_tensor(np.asarray(embedding),device=device);n,d=x.shape
    rng=np.random.RandomState(seed);norm=x.square().sum(1);best=float('inf');best_labels=None;best_centers=None
    tol=float(x.var(0,unbiased=False).mean())*1e-4
    def distances(centers):return (norm[:,None]+centers.square().sum(1)[None,:]-2*(x@centers.T)).clamp_min_(0)
    with torch.no_grad():
        for run in range(n_init):
            first=int(rng.choice(n));centers=x[first:first+1].clone();closest=distances(centers)[:,0]
            for _ in range(1,n_clusters):
                cumulative=closest.double().cumsum(0);pot=float(cumulative[-1]);trials=2+int(np.log(n_clusters))
                draws=torch.as_tensor(rng.uniform(size=trials)*pot,device=device)
                candidates=torch.searchsorted(cumulative,draws).clamp_max(n-1)
                costs=torch.minimum(closest[:,None],distances(x[candidates])).double().sum(0)
                picked=candidates[int(costs.argmin())]
                centers=torch.cat((centers,x[picked].unsqueeze(0)));closest=torch.minimum(closest,distances(centers[-1:])[:,0])
            previous=None
            for iteration in range(max_iter):
                ds=distances(centers);labels=ds.argmin(1)
                if previous is not None and torch.equal(labels,previous):break
                membership=torch.nn.functional.one_hot(labels,n_clusters).to(x.dtype)
                counts=membership.sum(0);new_centers=membership.T@x/counts.clamp_min(1)[:,None]
                if (counts==0).any():
                    empty=torch.where(counts==0)[0];far=ds.min(1).values.argsort(descending=True)[:len(empty)]
                    new_centers[empty]=x[far]
                shift=float((new_centers-centers).square().sum());centers=new_centers;previous=labels
                if shift<=tol:break
            ds=distances(centers);labels=ds.argmin(1);inertia=float(ds[torch.arange(n,device=device),labels].double().sum())
            if inertia<best:best=inertia;best_labels=labels.cpu().numpy().copy();best_centers=centers.cpu().numpy().copy()
            if report:report(dict(stage='KMeans',initialization=run+1,total_initializations=n_init,iterations=iteration+1,inertia=inertia,best_inertia=best))
    return SimpleNamespace(labels_=best_labels,cluster_centers_=best_centers,inertia_=best)


def cluster_embedding(export_directory, k, *, seed=40700, device='cuda:2', report=None):
    """Cluster the exact compact h exported by V1 GNN, with ten starts.

    K changes create only a clustering cache. This never loads/train models,
    decodes CCC, fits another projection, or changes the calibrated h vectors.
    """
    export_directory = Path(export_directory)
    metadata = json.loads((export_directory / 'export.json').read_text())
    family = metadata.get('backend')
    if family != 'gnn' or not metadata.get('global_training_affine_calibration'):
        raise ValueError('An identified, globally calibrated model export is required')
    h = np.load(export_directory / 'embedding.npy', mmap_mode='r')
    if not isinstance(k, int) or not 2 <= k <= len(h) or not np.isfinite(h).all():
        raise ValueError('Invalid K or nonfinite compact representation')
    protocol = dict(backend=family, export=metadata, embedding=array_identity(h),
                    k=k, seed=seed, n_init=10, row_normalization=False,
                    metric='Euclidean on fixed-global-training-calibrated h')
    folder = output_path(export_directory / 'clustering' / digest(protocol)[:16])
    if (folder / 'complete.json').exists():
        if json.loads((folder / 'complete.json').read_text())['protocol'] != protocol:
            raise ValueError('Incompatible saved clustering')
        return folder
    result = gpu_kmeans(h, k, seed=seed, n_init=10, device=device, report=report)
    np.save(folder / 'labels.npy', result.labels_.astype(np.int32))
    np.save(folder / 'centers.npy', result.cluster_centers_)
    write_json(folder / 'complete.json', dict(protocol=protocol, inertia=result.inertia_,
               cells=len(h), classes=int(len(np.unique(result.labels_))), model_retrained=False))
    return folder
