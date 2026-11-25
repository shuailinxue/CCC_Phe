"""Official scVI + Delaunay/three-hop aggregation + CellCharter GMM."""
from pathlib import Path
import json
import subprocess
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch


def cellcharter(a, graph, train_cells, train_centers, config, output):
    import cellcharter as cc
    import squidpy as sq
    output = Path(output)
    python = Path(config['cellcharter_scvi_python'])
    if not python.is_file():
        raise FileNotFoundError(f'Compatible scVI environment missing: {python}')
    np.save(output/'training_cells.npy',train_cells)
    cfg = output/'scvi_config.json'; cfg.write_text(json.dumps(config,indent=2))
    raw = output/'counts.h5ad'; a.write_h5ad(raw)
    if not (output/'scvi_audit.json').exists():
        with (output/'scvi_training.log').open('w') as log:
            result = subprocess.run([str(python),str(Path(__file__).with_name('scvi_runner.py')),
                                    str(raw),str(output),str(cfg)],stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            tail = '\n'.join((output/'scvi_training.log').read_text().splitlines()[-30:])
            raise RuntimeError(f'scVI failed; log: {output / "scvi_training.log"}\n{tail}')
    full = a.copy(); full.obs['sample_id'] = pd.Categorical(full.obs.sample_id)
    full.obsm['X_scVI'] = np.load(output/'scvi_latent.npy')
    # Official transcriptomics tutorial graph; build within each sample.
    sq.gr.spatial_neighbors(full,library_key='sample_id',coord_type='generic',
                           delaunay=True,spatial_key='spatial',percentile=99)
    connectivity = full.obsp['spatial_connectivities'].tocsr()
    sp.save_npz(output/'spatial_connectivities.npz',connectivity)
    layers = config['cellcharter_layers']
    cc.gr.aggregate_neighbors(full,n_layers=layers,use_rep='X_scVI',sample_key='sample_id')
    tr = full[train_cells].copy()
    cc.gr.aggregate_neighbors(tr,n_layers=layers,use_rep='X_scVI',sample_key='sample_id')
    positions = np.searchsorted(train_cells,train_centers)
    if not np.array_equal(train_cells[positions],train_centers):
        raise ValueError('Training center alignment error')
    fitting = tr[positions].copy()
    trainer = {'accelerator':'gpu','devices':[int(config['device'].split(':')[1])]} if config['device'].startswith('cuda:') else {'accelerator':'cpu','devices':1}
    trainer.update(max_epochs=config['gmm_epochs'],logger=False,enable_progress_bar=False)
    model = cc.tl.Cluster(n_clusters=config['fixed_k'],random_state=config['seed'],trainer_params=trainer)
    model.fit(fitting,use_rep='X_cellcharter')
    # Exact fitted native TorchGMM posterior; bypass old Lightning loader only.
    mixture = model.model_.to(config['device']); mixture.eval()
    with torch.no_grad():
        labels = np.concatenate([mixture(torch.as_tensor(v,device=config['device']))[0].argmax(-1).cpu().numpy()
                                 for v in np.array_split(full.obsm['X_cellcharter'],max(1,int(np.ceil(len(full)/2048))))])
    model.save(output/'gmm')
    (output/'adapter_audit.json').write_text(json.dumps({
        'variant':'Official CellCharter transcriptomics: scVI + Delaunay + GMM',
        'official_tutorial':'https://cellcharter.readthedocs.io/en/latest/notebooks/cosmx_human_nsclc.html',
        'n_layers':layers,'graph':'Per-sample Delaunay; percentile=99',
        'native_scvi_and_GMM_models_unchanged':True,
        'truth_MI_labels_or_loading_used':False,
        'protocol_adaptations':['Train-only scVI and induced training graph','Fixed K=4 shared comparison','Public fitted TorchGMM posterior prediction']},indent=2)+'\n')
    return np.asarray(full.obsm['X_cellcharter'],np.float32),np.asarray(fitting.obsm['X_cellcharter'],np.float32),labels.astype(int)


def safe_test_rows(connectivity, owner, layers):
    """Only retain centers whose actual native aggregation donors share fold."""
    result = np.ones(len(owner),bool)
    for i in range(len(owner)):
        reach = {i}; frontier = {i}
        for _ in range(layers):
            next_nodes = set()
            for j in frontier:
                next_nodes.update(connectivity.indices[connectivity.indptr[j]:connectivity.indptr[j+1]])
            frontier = next_nodes-reach; reach.update(next_nodes)
        result[i] = owner[i]>=0 and np.all(owner[list(reach)]==owner[i])
    return result
