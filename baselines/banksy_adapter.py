"""Official BANKSY mean/AGF feature construction with training-only PCA/scaling."""
from pathlib import Path
import json
import joblib
import numpy as np
from scipy.sparse import diags,save_npz
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits
from .adapters import log_expression
from .protocol import isolate_graph


def banksy(a, train_cells, owner, config, output):
    from .vendor.banksy.main import generate_spatial_weights_fixed_nbrs,concatenate_all
    from .vendor.banksy.embed_banksy import create_nbr_matrix
    output=Path(output); full=a.copy();full.X=log_expression(a)
    xy=np.asarray(a.obsm['spatial']);weights={};training_weights={}
    safe=owner>=0
    for m in (0,1):
        k=min(config['banksy_k_geom'],(len(a)-1)//(m+1))
        w,_,_=generate_spatial_weights_fixed_nbrs(xy,m=m,num_neighbours=k,
                        decay_type=config['banksy_weight_decay'],verbose=False)
        support=w.tocoo();bad=np.unique(support.row[owner[support.row]!=owner[support.col]])
        safe[bad]=False
        weights[m]=w
        save_npz(output/f'weights_m{m}.npz',w)
        w=isolate_graph(w,owner)
        # The native complex AGF has exp(i*m*theta) times positive radial
        # weights. Re-normalize magnitudes after removing cross-fold donors.
        norm=np.asarray(abs(w).sum(1)).ravel()
        w=(diags(1/np.maximum(norm,1e-12))@w).tocsr();training_weights[m]=w
        save_npz(output/f'training_weights_m{m}.npz',w)
    np.save(output/'inference_safe_centers.npy',safe)
    d={config['banksy_weight_decay']:{'weights':weights}}
    with threadpool_limits(limits=config.get('cpu_threads',6)):
        neighbor=create_nbr_matrix(full,d,config['banksy_weight_decay'],1,
                                   center=True,variance_balance=False,verbose=False)
        blocks=[full.X,neighbor[0],neighbor[1]]
        td={config['banksy_weight_decay']:{'weights':training_weights}}
        train_neighbor=create_nbr_matrix(full,td,config['banksy_weight_decay'],1,
                                         center=True,variance_balance=False,verbose=False)
        train_blocks=[full.X[train_cells],train_neighbor[0][train_cells],train_neighbor[1][train_cells]]
        combined=[np.concatenate((tr,all_rows),axis=0) for tr,all_rows in zip(train_blocks,blocks)]
        augmented=concatenate_all(combined,config['banksy_lambda'],reference_rows=np.arange(len(train_cells)))
        augmented=np.asarray(augmented,np.float32)
        if not np.isfinite(augmented).all():raise ValueError('Invalid native BANKSY feature matrix')
        pca=PCA(n_components=min(config['pca_components'],len(train_cells)-1,augmented.shape[1]),
                svd_solver='randomized',random_state=config['seed']).fit(augmented[:len(train_cells)])
        training=pca.transform(augmented[:len(train_cells)]).astype(np.float32)
        embedding=pca.transform(augmented[len(train_cells):]).astype(np.float32)
    joblib.dump(pca,output/'pca.joblib')
    np.savez(output/'preprocessing.npz',training_cells=train_cells,
             block_means=np.stack([np.asarray(v).mean(0) for v in train_blocks]),
             block_variances=np.stack([np.asarray(v).var(0) for v in train_blocks]),
             block_valid=np.stack([np.asarray(v,np.float64).std(0)>1e-8 for v in train_blocks]))
    (output/'adapter_audit.json').write_text(json.dumps({
        'variant':'Official BANKSY Python spatial features, mean + centered m=1 AGF, training-only PCA',
        'native_kernel_AGF_and_lambda_weights_unchanged':True,
        'lambda':config['banksy_lambda'],'k_geom':config['banksy_k_geom'],
        'physical_neighbor_counts':[config['banksy_k_geom'],2*config['banksy_k_geom']],
        'weight_decay':config['banksy_weight_decay'],'cell_type_or_CCC_input':False,
        'truth_MI_labels_or_loading_used':False,'normalization_and_PCA_fit_on_training_only':True,
        'fold_safety':'Fit on isolated training graph; frozen scaling/PCA applied to full native inference graph. Common test also excludes centers whose actual BANKSY donors cross physical folds.',
        'clustering':'Predeclared common K=4; BANKSY supports KMeans, native Leiden not tuned using truth'
    },indent=2)+'\n')
    return embedding,training
