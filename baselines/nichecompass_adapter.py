"""Official NicheCompass with public LR masks and native de novo programs."""
from pathlib import Path
import json
import numpy as np
import torch
from .protocol import panel_lr_prior


def nichecompass(a, graph, train_cells, config, output):
    # DGL/PyG can load separate BLAS/OpenMP runtimes. Bound every loaded
    # pool after the official import; otherwise tiny batches oversubscribe
    # a many-core server despite PyTorch's own thread setting.
    import nichecompass as official_package
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=config.get('cpu_threads',6)):
        return _fit(a,graph,train_cells,config,output)


def _fit(a, graph, train_cells, config, output):
    from nichecompass.models import NicheCompass
    from nichecompass.utils import add_gps_from_gp_dict_to_adata
    output = Path(output)
    print('NicheCompass: official imports ready',flush=True)
    if config['device'].startswith('cuda:'):
        torch.cuda.set_device(int(config['device'].split(':')[1]))
    prior = panel_lr_prior(a.var_names, output)
    programs = {}
    for ligand, links in prior.groupby('ligand', sort=True):
        targets = sorted(set(links.receptor))
        programs[f'PUBLIC_{ligand}_LR'] = {'sources':[ligand], 'sources_categories':['ligand'],
                                         'targets':targets, 'targets_categories':['receptor']*len(targets)}
    full = a.copy(); full.obsp['spatial_connectivities'] = graph.maximum(graph.T).tocsr()
    add_gps_from_gp_dict_to_adata(programs, full, min_source_genes_per_gp=1, min_target_genes_per_gp=1,
                                 filter_genes_not_in_masks=False)
    print(f'NicheCompass: {len(programs)} public prior programs; constructing model',flush=True)
    tr = full[train_cells].copy()
    reuse = output/'reuse_training.json'
    if reuse.exists():
        proof=json.loads(reuse.read_text())
        original={k:v for k,v in proof['manifest']['config'].items() if k!='result_subdir'}
        current={k:v for k,v in config.items() if k!='result_subdir'}
        if original!=current:raise ValueError('NicheCompass checkpoint training configuration changed')
        from phenoniche.simulation.response_simulation import checksum
        for path,digest in proof['checkpoint_sha256'].items():
            if checksum(output/path)!=digest:raise ValueError('Reused NicheCompass checkpoint changed')
        model=NicheCompass.load(str(output/'model'),adata=tr,use_cuda=config['device']!='cpu')
        print('NicheCompass: reusing identical trained weights; native active-program inference',flush=True)
    else:
        model = NicheCompass(tr, counts_key=None, adj_key='spatial_connectivities',
                         n_addon_gp=config['nichecompass_addon_programs'],
                         n_hidden_encoder=config['nichecompass_hidden'],
                         include_chrom_access_recon_loss=False, use_cuda_if_available=config['device']!='cpu',seed=config['seed'])
        print('NicheCompass: fitting official model',flush=True)
        model.train(n_epochs=config['nichecompass_epochs'], n_epochs_all_gps=min(25,config['nichecompass_epochs']),
                node_batch_size=config['nichecompass_node_batch'],edge_batch_size=config.get('nichecompass_edge_batch',config['nichecompass_node_batch']),
                lr=config['nichecompass_lr'], use_cuda_if_available=config['device']!='cpu',
                use_early_stopping=False, monitor=True, verbose=False)
    print('NicheCompass: inferring saved embeddings',flush=True)
    embedding = model.get_latent_representation(adata=full,counts_key=None, only_active_gps=True,
                                                node_batch_size=config['nichecompass_node_batch'],dtype=np.float32)
    training = model.get_latent_representation(adata=tr,counts_key=None,only_active_gps=True,
                                               node_batch_size=config['nichecompass_node_batch'],dtype=np.float32)
    if not reuse.exists():model.save(str(output / 'model'), overwrite=True, save_adata=False)
    np.save(output/'training_embedding.npy',training)
    import pandas as pd
    if not reuse.exists():pd.DataFrame(dict(model.trainer.epoch_logs)).to_csv(output/'training.csv',index=False)
    (output/'adapter_audit.json').write_text(json.dumps({'variant':'Official NicheCompass, public LR-prior + de novo programs',
        'prior_source':'Full commuspace_human_lr_atlas.tsv (NicheNet v2, CellChat v2, LIANA); panel intersection only',
        'matched_LR_pairs':len(prior), 'prior_programs':len(programs), 'addon_programs':config['nichecompass_addon_programs'],
        'clustering_representation':'Official active-program posterior, only_active_gps=True',
        'active_programs':embedding.shape[1],
        'synthetic_background_genes_have_no_fabricated_priors':True,'cell_type_input':False,
        'truth_MI_labels_or_loading_used':False,'training_physical_cells_only':True,
        'limitation':'LR-only prior variant; synthetic genes have no curated target network. No full NicheNet/TF prior is claimed.'},indent=2)+'\n')
    return embedding,training
