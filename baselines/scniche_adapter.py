"""Official scNiche three-view MGAE/graph fusion/MMIM with fold-safe batches."""
from pathlib import Path
import json
import joblib
import anndata as ad
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.decomposition import PCA

from .adapters import log_expression
from .protocol import isolate_graph, row_normalize


def scniche(a, graph, train_cells, types, categories, owner, config, output):
    from .vendor.scniche.scniche.preprocess._utils import construct_graph
    from .vendor.scniche.scniche.trainer._train import Runner_batch
    output = Path(output)
    torch.manual_seed(config['seed']); np.random.seed(config['seed'])
    g = isolate_graph(graph, owner)
    x = log_expression(a)
    mean = np.asarray(row_normalize(g) @ x, np.float32)
    alone = np.asarray(g.sum(1)).ravel() == 0
    mean[alone] = x[alone]
    # Native features: center composition includes self; neighbor expression
    # excludes self. Both PCAs are fit strictly on training physical cells.
    onehot = np.eye(len(categories), dtype=np.float32)[types]
    comp = row_normalize(g + sp.eye(len(a), format='csr')) @ onehot
    dim = min(config['pca_components'], len(train_cells) - 1, a.n_vars)
    own_pca = PCA(dim, random_state=config['seed']).fit(x[train_cells])
    nbr_pca = PCA(dim, random_state=config['seed']).fit(mean[train_cells])
    features = [np.asarray(comp, np.float32), own_pca.transform(x).astype(np.float32), nbr_pca.transform(mean).astype(np.float32)]
    keys = ['X_cn_norm', 'X_data', 'X_data_nbr']
    size = min(config['scniche_batch_size'], len(train_cells))
    if size < 4:
        raise ValueError('Insufficient scNiche training cells')
    rng = np.random.default_rng(config['seed'])

    def batches(cells):
        cells = rng.permutation(cells)
        result = []
        for start in range(0, len(cells), size):
            ids = cells[start:start + size]
            if len(ids) < size:
                ids = np.r_[ids, rng.choice(cells, size - len(ids), replace=True)]
            result.append(ids)
        return result

    def graphs(ids):
        return [construct_graph(v[ids], min(config['scniche_graph_k'], size - 1), min(config['scniche_mik'], size)) for v in features]

    fitting = ad.AnnData(sp.csr_matrix((len(train_cells), 1)))
    for key, values in zip(keys, features):
        fitting.obsm[key] = values[train_cells]
    train_batches = batches(train_cells)
    fitting.uns['batch_idx'] = [list(np.searchsorted(train_cells, ids)) for ids in train_batches]
    # A reusable list of official DGL view graphs has the same iteration
    # interface as the native GraphDataLoader without changing the loss.
    fitting.uns['dataloader'] = [graphs(ids) for ids in train_batches]
    runner = Runner_batch(fitting, choose_views=keys, device=config['device'], verbose=False)
    runner.fit(lr=config['scniche_lr'], epochs=config['scniche_epochs'])
    model = runner.model; model.eval()
    embedding = np.empty((len(a), runner.hidden_size[-1]), np.float32)
    torch.manual_seed(config['seed'] + 1)
    with torch.no_grad():
        # Graph fusion has node-count-dependent weights. Reuse the exact
        # training batch size, pad within the same fold, and discard padding.
        # Official dropout remains as released; fixed seeds record its draws.
        for fold in sorted(np.unique(owner)):
            cells = np.flatnonzero(owner == fold)
            for ids in batches(cells):
                gs = [v.to(config['device']) for v in graphs(ids)]
                _, _, z = model(gs, [v.ndata['feat'] for v in gs], config['device'])
                unique, first = np.unique(ids, return_index=True)
                embedding[unique] = z.detach().cpu().numpy()[first]
    torch.save({'model_state':model.state_dict(), 'discriminator_state':runner.model_d.state_dict(),
                'batch_size':size, 'train_batches':train_batches, 'training_cells':train_cells,
                'config':config, 'hidden_size_v':runner.hidden_size_v, 'hidden_size':runner.hidden_size}, output / 'checkpoint.pt')
    joblib.dump({'own_pca':own_pca,'neighbor_pca':nbr_pca,'categories':categories}, output / 'preprocessing.joblib')
    import pandas as pd
    pd.DataFrame({'epoch':np.arange(1,len(fitting.uns['loss'])+1), 'native_loss':np.asarray(fitting.uns['loss'],float)}).to_csv(output / 'training.csv',index=False)
    (output / 'adapter_audit.json').write_text(json.dumps({'variant':'Official three-view Runner_batch, PCA input',
        'physical_fold_isolated_views_and_batches':True, 'native_MGAE_GFN_MMIM_unchanged':True,
        'cell_type_input':True, 'truth_MI_labels_or_loading_used':False, 'batch_size':size},indent=2)+'\n')
    return embedding, embedding[train_cells]
