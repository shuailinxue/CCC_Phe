"""Training-RNA-only PCA cell states, preserving expression-only encoder inputs."""
from pathlib import Path
import hashlib,json
import numpy as np
from sklearn.decomposition import PCA
from .data import expression_block


def prepare_cell_states(data,config,cache,projection_cells=None):
    if not config.cell_state_components:return
    cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
    training_cells=np.unique(data.members[data.rows[0]])
    sample=np.random.default_rng(config.split_seed).choice(training_cells,min(20000,len(training_cells)),replace=False)
    if len(sample)<=config.cell_state_components:raise ValueError('Insufficient training cells for RNA-only PCA')
    signature={'source':data.audit['source_signature'],'cell_order':data.audit['cell_order_sha256'],
        'genes':list(data.genes),'training_cells_sha256':hashlib.sha256(training_cells.tobytes()).hexdigest(),
        'fit_sample_sha256':hashlib.sha256(sample.tobytes()).hexdigest(),'components':config.cell_state_components,
        'seed':config.split_seed,'normalization':'all-panel library-normalized log expression; training gene z-score; whitened PCA',
        'cell_type_or_CCC_inputs':False}
    parameters=cache/'cell_state_pca.npz';metadata=cache/'cell_state_pca.json'
    if parameters.exists() or metadata.exists():
        if not parameters.exists() or not metadata.exists() or json.loads(metadata.read_text())['signature']!=signature:raise ValueError('Existing RNA cell-state cache differs or is incomplete; protected from overwrite')
        with np.load(parameters) as values:pre={key:values[key].copy() for key in values.files}
    else:
        all_genes=np.arange(len(data.genes));x=expression_block(data.expression,sample,all_genes,data.totals)
        gene_mean=x.mean(0);gene_scale=np.maximum(x.std(0),.1)
        x=(x-gene_mean)/gene_scale
        pca=PCA(n_components=config.cell_state_components,svd_solver='randomized',random_state=config.split_seed,whiten=True,n_oversamples=16,iterated_power=4).fit(x)
        pre={'gene_mean':gene_mean,'gene_scale':gene_scale,'pca_mean':pca.mean_,'components':pca.components_,
            'whitening':np.maximum(np.sqrt(pca.explained_variance_),1e-6),'explained_variance_ratio':pca.explained_variance_ratio_,'fit_cell_ids':sample}
        np.savez(parameters,**pre)
        metadata.write_text(json.dumps({'signature':signature,'training_physical_cells':len(training_cells),'fit_cells':len(sample),
            'panel_genes':len(data.genes),'explained_variance_fraction':float(pca.explained_variance_ratio_.sum())},indent=2)+'\n')
    scores_path=cache/'cell_state_scores.npy';shape=(len(data.cells),config.cell_state_components)
    if scores_path.exists():
        scores=np.load(scores_path,mmap_mode='r+')
        if scores.shape!=shape:raise ValueError('RNA PCA score cache shape mismatch')
    else:
        scores=np.lib.format.open_memmap(scores_path,mode='w+',dtype=np.float32,shape=shape);scores[:]=np.nan;scores.flush()
    requested=np.arange(len(data.cells)) if projection_cells is None else np.unique(np.asarray(projection_cells,dtype=np.int64))
    all_genes=np.arange(len(data.genes))
    for start in range(0,len(requested),4096):
        rows=requested[start:start+4096];rows=rows[~np.isfinite(scores[rows]).all(1)]
        if len(rows)==0:continue
        x=expression_block(data.expression,rows,all_genes,data.totals)
        x=(x-pre['gene_mean'])/pre['gene_scale']-pre['pca_mean']
        scores[rows]=(x@pre['components'].T)/pre['whitening']
    scores.flush();data.cell_state_scores=scores;data.cell_state_preprocessing=pre
    data.audit['cell_state_RNA_PCA']={'components':config.cell_state_components,'panel_genes':len(data.genes),
        'fit_cells':len(sample),'fit':'training physical cells only, no cell-type/CCC labels',
        'projected_cells':len(requested),'expression_features':len(data.encoder_indices)+config.cell_state_components+1,
        'explained_variance_fraction':float(pre['explained_variance_ratio'].sum())}
    print(f'Expression-only PCA{config.cell_state_components}: fitted {len(sample)} training cells; projected {len(requested)} cells',flush=True)
