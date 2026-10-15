"""Official SpaNiche/IntNMF R solver with a single-cell, fold-safe adapter.

The published multiscale abundance/LR product definitions are retained.
Raw cell types replace spot deconvolution; radii are derived from the common
training KNN scale, rather than applying Visium spot units to imaging cells.
"""
from pathlib import Path
import json
import subprocess
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import cKDTree
from .adapters import log_expression
from .protocol import panel_lr_prior, row_normalize


def radius_views(coordinates, owner, radius):
    n = len(coordinates)
    first, second, annulus = [], [], []
    tree = cKDTree(coordinates)
    for i, ids in enumerate(tree.query_ball_point(coordinates, 2 * radius)):
        ids = np.asarray(ids, np.int64)
        ids = ids[owner[ids] == owner[i]]
        d = np.linalg.norm(coordinates[ids] - coordinates[i], axis=1)
        near = ids[d <= radius]
        far = ids[d > radius]
        first.extend((i, int(j)) for j in near)
        second.extend((i, int(j)) for j in ids)
        # Official LR view2 is the outer annulus PLUS the central cell.
        annulus.extend((i, int(j)) for j in np.r_[i, far])
    def matrix(pairs):
        row, col = np.asarray(pairs).T
        return row_normalize(sp.csr_matrix((np.ones(len(row)), (row, col)), shape=(n,n)))
    return matrix(first), matrix(second), matrix(annulus)


def spaniche(a, members, train_cells, types, categories, owner, config, output):
    output = Path(output)
    prior = panel_lr_prior(a.var_names, output)
    xy = np.asarray(a.obsm['spatial'], np.float64)
    # Scale derived from training geometry only: median farthest KNN30
    # physical distance. Two views use radius and 2*radius in micrometres.
    radius = float(np.median(np.linalg.norm(xy[members[train_cells,-1]] - xy[train_cells],axis=1)))
    if radius <= 0:
        raise ValueError('Invalid SpaNiche physical neighborhood scale')
    near, far, annulus = radius_views(xy, owner, radius)
    onehot = np.eye(len(categories))[types]
    abundance = np.column_stack((onehot, near @ onehot, far @ onehot))
    counts = log_expression(a)
    lookup = {g:i for i,g in enumerate(a.var_names)}
    ligand = counts[:, [lookup[g] for g in prior.ligand]]
    receptor = counts[:, [lookup[g] for g in prior.receptor]]
    # These products equal official generate_LRintegratedmatrix(): center
    # ligand × mean neighboring receptor, and the converse, at two scales.
    lr = np.column_stack((ligand*(near@receptor), receptor*(near@ligand),
                          ligand*(annulus@receptor), receptor*(annulus@ligand)))
    variance = lr[train_cells].var(0, ddof=1)
    lo, hi = np.quantile(variance, config['spaniche_variance_quantiles'])
    mask = (variance > lo) & (variance < hi) & (lr[train_cells].sum(0) > 0)
    if not mask.any():
        raise ValueError('SpaNiche training-only LR variance filter retains no features')
    np.save(output/'LR_feature_mask.npy',mask)
    pd.DataFrame(abundance).to_csv(output/'abundance.csv',index=False)
    pd.DataFrame(lr[:,mask]).to_csv(output/'LR_views.csv',index=False)
    pd.DataFrame(xy,columns=['row','col']).to_csv(output/'coordinates.csv',index=False)
    pd.DataFrame({'owner':owner}).to_csv(output/'physical_owner.csv',index=False)
    settings = {'seed':config['seed'],'rank':config['spaniche_topics'], 'lambda':config['spaniche_lambda'],
                'sigma':config['spaniche_sigma_um'],'maxiter':config['spaniche_maxiter'],
                'initial_maxiter':config['spaniche_initial_maxiter'],'initial_restarts':config['spaniche_initial_restarts'],
                'st_count':20,'epsilon':1e-4}
    (output/'solver_settings.json').write_text(json.dumps(settings,indent=2)+'\n')
    args = [str(output), str(settings['seed']),str(settings['rank']),str(settings['lambda']),str(settings['sigma']),
            str(settings['maxiter']),str(settings['initial_maxiter']),str(settings['initial_restarts'])]
    with (output/'R_execution.log').open('w') as log:
        subprocess.run([config['rscript'],str(Path(__file__).with_name('spaniche_runner.R')),*args],stdout=log,stderr=subprocess.STDOUT,check=True)
    embedding = pd.read_csv(output/'W.csv').to_numpy(np.float32)
    if embedding.shape != (len(a),settings['rank']):
        raise ValueError('SpaNiche factor/cell-order mismatch')
    (output/'adapter_audit.json').write_text(json.dumps({'variant':'Official SpaNiche joint abundance/LR graph NMF + official IntNMF initialization',
        'single_cell_adapter':'Observed one-hot types replace spot deconvolution; no inferred anatomical labels',
        'views':'Abundance: self, radius, 2*radius. LR: center-L/neighbor-R and converse; inner disc / outer annulus+self.',
        'radii_um':[radius,2*radius],'graph_sigma_um':settings['sigma'],'public_LR_pairs':len(prior),
        'retained_LR_view_features':int(mask.sum()),'variance_filter_fit_on_training_only':True,
        'native_factorization_unchanged':True,'heldout_inference':'Fixed training H/weights, native NNLS then W-only graph updates within each held-out physical fold',
        'cell_type_input':True,'truth_MI_labels_or_loading_used':False,
        'fair_clustering':'Fixed K=4 on W, fit on original training centers. Native topic-argmax labels saved separately.'},indent=2)+'\n')
    return embedding,embedding[train_cells]
