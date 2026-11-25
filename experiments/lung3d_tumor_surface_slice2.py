"""Exploratory 3D communication programs for section 2 tumor-surface cells."""
import os, pickle
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.cluster import KMeans
from phenoniche.spatial import Settings
from phenoniche.spatial.channels import Channels
from phenoniche.spatial.cell_programs import CellTensor
from phenoniche.spatial.programs import fit_programs, project
import lung3d_v2 as study

out = Path('/data1/xueshuailin/CCC_Phe/workspace/outputs/3D_lung_CosMx/v2_bidirectional/tumor_surface_slice2')
out.mkdir(parents=True, exist_ok=True)
a, xyz = study.read_input()
z = np.unique(xyz[:, 2])[1]
ids = np.flatnonzero((xyz[:, 2] == z) & a.obs.niches_3D.eq('Tumor surface').to_numpy())
print(f'Second section z={z}, tumor surface cells: {len(ids)}', flush=True)
_, types = np.unique(a.obs.celltypes.astype(str), return_inverse=True)
graph = load_npz(study.OUT/'contacts.npz')
channels = Channels(a.layers['counts'], a.var_names, pd.read_csv(study.ROOT/'data/commuspace_human_lr_atlas.tsv', sep='\t'))
cfg = replace(Settings(), fit_windows=512, iterations=40)
tensor = CellTensor(graph, types, channels, cfg.sigma_um)
allowed = np.ones(a.n_obs, dtype=bool)
rng = np.random.default_rng(73)
train = rng.choice(ids, min(cfg.fit_windows, len(ids)), replace=False)
print('Fitting CP', flush=True)
model = fit_programs([tensor.fiber(i, allowed) for i in train], tensor.n_types, cfg, rng, n_senders=1)
if model is None: raise RuntimeError('No usable communication signal')
scores = np.zeros((len(ids), model[2].shape[1]), dtype=np.float32)
for j, i in enumerate(ids):
    scores[j] = np.nan_to_num(project(tensor.fiber(i, allowed), model))
    if (j+1)%2000==0: print(f'Projected {j+1}/{len(ids)}', flush=True)
scale = np.maximum(np.quantile(scores, .9, axis=0), 1e-7)
features = np.log1p(scores / scale)
results = {'cell_id': np.asarray(a.obs_names[ids]), 'rank': scores.shape[1], 'z': z}
for k in (2,3,4):
    results[f'k{k}'] = KMeans(n_clusters=k, n_init=10, random_state=73).fit_predict(features)
with (out/'result.pkl').open('wb') as f: pickle.dump({'results':results,'model':model,'scores':scores}, f, protocol=5)
print('Finished', flush=True)
