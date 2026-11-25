"""Exploratory single-seed fusion: Harmonics composition and NicheCompass h.

The source of K is the existing brain benchmark. Reference labels are loaded
only after fitting; this script does not change either method or V3.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '3')
os.environ.setdefault('OMP_NUM_THREADS', '3')
os.environ.setdefault('MKL_NUM_THREADS', '3')
import sys, json, pickle, random, time
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

W = Path('/data1/xueshuailin/CCC_Phe/workspace')
BASE = W / 'data/harmonics_fig2'
OUT = W / 'outputs/brain_harmonics_nichecompass_fusion'
BENCH = Path('/home/nas3/biod/xueshuailin/CCC_Phe/workspace/outputs/brain_10run_benchmark')
sys.path.insert(0, '/home/nas3/biod/xueshuailin/CCC_Phe/workspace/artifacts/baselines/Harmonics')
from Harmonics import Harmonics_Model
from nichecompass.models import NicheCompass
from nichecompass.utils import add_gps_from_gp_dict_to_adata

dataset = sys.argv[1]
assert dataset in ('starmap', 'merfish')
seed, k = 112, 7
torch.set_num_threads(3)
torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
torch.cuda.set_device(0)
out = OUT / dataset
out.mkdir(parents=True, exist_ok=True)
path = BASE / ('starmap/analysis_input.h5ad' if dataset == 'starmap' else 'merfish_scaled/donor12_slice0_analysis_input.h5ad')
source = ad.read_h5ad(path)
xy = np.asarray(source.obsm['spatial']).copy()
types = source.obs.cell_types.astype(str).to_numpy()
obs = pd.DataFrame({'celltype': types}, index=source.obs_names.copy())
counts = sparse.csr_matrix(source.layers['counts'], dtype=np.float32)
a = ad.AnnData(X=counts.copy(), obs=obs, var=pd.DataFrame(index=source.var_names.copy()))
a.layers['counts'] = counts
a.obsm['spatial'] = xy
del source
print('START', dataset, 'shape', a.shape, 'seed', seed, flush=True)

# Official Harmonics local composition: same settings as the simulation fusion.
hm = Harmonics_Model([a.copy()], ['slice'], seed=seed, parallel=False, verbose=False, light_mode=True)
hm.preprocess(ct_key='celltype', spatial_key='spatial', method='joint', n_step=3, n_neighbors=20)
p = hm.micro_dist_basic.toarray() if sparse.issparse(hm.micro_dist_basic) else np.asarray(hm.micro_dist_basic)
np.save(out / 'composition.npy', p.astype('float32'))
del hm
print('COMPOSITION', p.shape, flush=True)

prior_path = W / 'data/priors/nichecompass_original_mouse/public_mouse_combined.pkl'
gp = pickle.load(prior_path.open('rb'))
ind = cKDTree(xy).query(xy, k=5, workers=3)[1]
g = sparse.csr_matrix((np.ones(len(a)*4, dtype='float32'), (np.repeat(np.arange(len(a)), 4), ind[:, 1:].ravel())), shape=(len(a), len(a)))
a.obsp['spatial_connectivities'] = g.maximum(g.T).tocsr()
add_gps_from_gp_dict_to_adata(gp, a, genes_uppercase=True, min_genes_per_gp=1, min_source_genes_per_gp=0, min_target_genes_per_gp=0, filter_genes_not_in_masks=False)
print('GP preparation completed; uns keys', list(a.uns.keys()), flush=True)
t = time.monotonic()
model = NicheCompass(a, counts_key='counts', adj_key='spatial_connectivities', n_addon_gp=10, active_gp_thresh_ratio=.05, gene_expr_recon_dist='nb', n_fc_layers_encoder=1, conv_layer_encoder='gcnconv', n_hidden_encoder=None, log_variational=True, node_label_method='one-hop-norm', use_cuda_if_available=True, seed=seed)
print('TRAINING 400 epochs', flush=True)
model.train(n_epochs=400, n_epochs_all_gps=25, n_epochs_no_cat_covariates_contrastive=5, lr=.001, lambda_edge_recon=5000000., lambda_gene_expr_recon=3000., lambda_cat_covariates_contrastive=0., contrastive_logits_pos_ratio=0., contrastive_logits_neg_ratio=0., lambda_group_lasso=0., lambda_l1_masked=0., lambda_l1_addon=0., edge_batch_size=256, node_batch_size=None, n_sampled_neighbors=-1, seed=seed, verbose=True, use_cuda_if_available=True)
h = model.get_latent_representation(adata=a, counts_key='counts', only_active_gps=True, node_batch_size=512, dtype=np.float32)
np.save(out / 'nichecompass_h.npy', h)
print('LATENT', h.shape, 'seconds', round(time.monotonic()-t), flush=True)

def norm(x):
    x = np.asarray(x, dtype=np.float32)
    x = x - x.mean(0)
    return x / max(float(np.sqrt(np.mean(np.sum(x*x, axis=1)))), 1e-8)

x = np.column_stack((norm(np.sqrt(np.maximum(p, 0))), norm(h)))
pred = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(x)
comp_pred = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(norm(np.sqrt(np.maximum(p, 0))))
nc_pred = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(norm(h))
np.savez_compressed(out / 'predictions.npz', fusion=pred, composition=comp_pred, nichecompass_kmeans=nc_pred, cell_ids=np.asarray(a.obs_names.astype(str)))

# Evaluation labels are read only after all predictions have been frozen.
ref = ad.read_h5ad(path)
if dataset == 'starmap':
    r = ad.read_h5ad(BASE / 'starmap/stDyer_mouse_cortex.h5ad')
    dd, ix = cKDTree(np.c_[r.obs.X, -r.obs.Y]).query(xy)
    valid = dd < .01
    truth = r.obs.label.iloc[ix].astype(str).str.upper().to_numpy()
else:
    valid = np.ones(len(ref), bool)
    truth = ref.obs.benchmark_region.astype(str).to_numpy()
result = {'dataset': dataset, 'seed': seed, 'K': k, 'n_total': len(a), 'n_evaluated': int(valid.sum()), 'nichecompass_h_dim': h.shape[1], 'composition_dim': p.shape[1], 'fusion_ARI': float(adjusted_rand_score(truth[valid], pred[valid])), 'composition_kmeans_ARI': float(adjusted_rand_score(truth[valid], comp_pred[valid])), 'nichecompass_kmeans_ARI': float(adjusted_rand_score(truth[valid], nc_pred[valid])), 'truth_use': 'evaluation only', 'fusion': 'sqrt composition and NicheCompass h, separately mean-centered and RMS-scaled, concatenated; KMeans n_init=10', 'nc_settings': 'exploratory transfer of prior simulation settings: mouse GP, symmetrized KNN4, 400 epochs; not claimed author-optimized', 'seconds': time.monotonic()-t}
for name in ('harmonics', 'v3'):
    f = BENCH / dataset / name / f'seed_{seed}' / 'metrics.json'
    if f.exists(): result[name + '_ARI_seed112'] = json.loads(f.read_text())['ARI']
(out / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
print('RESULT', json.dumps(result), flush=True)
