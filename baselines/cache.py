"""Reuse unchanged methods when extending a benchmark, preserving older reports."""
from pathlib import Path
import json
import shutil
from phenoniche.simulation.response_simulation import checksum

COMMON = ('device','seed','fixed_k','cpu_threads')
METHOD_CONFIG = {
    'graphst':('graphst_epochs','graphst_hidden'),
    'cellcharter':('cellcharter_representation','pca_components','cellcharter_layers','gmm_epochs','cellcharter_scvi_epochs','cellcharter_scvi_python'),
    'scniche':('pca_components','scniche_epochs','scniche_lr','scniche_batch_size','scniche_graph_k','scniche_mik'),
    'nichecompass':('nichecompass_epochs','nichecompass_addon_programs','nichecompass_hidden','nichecompass_node_batch','nichecompass_lr','nichecompass_edge_batch'),
    'spaniche':('spaniche_topics','spaniche_lambda','spaniche_sigma_um','spaniche_maxiter','spaniche_initial_maxiter','spaniche_initial_restarts','spaniche_variance_quantiles','rscript'),
    'stagate':('stagate_hidden_dims','stagate_epochs','stagate_lr','stagate_gradient_clipping','stagate_weight_decay'),
    'banksy':('pca_components','banksy_lambda','banksy_k_geom','banksy_weight_decay'),
}


def reuse_completed(previous, output, manifest):
    previous,output=Path(previous),Path(output)
    if previous.resolve()==output.resolve():raise ValueError('Reuse source must differ from new report')
    if not (previous/'complete.json').exists():return []
    old=json.loads((previous/'config.json').read_text())
    if old['inputs']!=manifest['inputs']:raise ValueError('Cannot reuse baseline models from different expression/CCC/cell inputs')
    if any(manifest['versions'].get(k)!=v for k,v in old['versions'].items()):
        raise ValueError('Cannot reuse baselines across changed package versions')
    reused=[]
    for method,keys in METHOD_CONFIG.items():
        source=previous/method;target=output/method
        if target.exists() or not (source/'complete.json').exists():continue
        if method=='graphst' and old.get('GraphST_source')!=manifest.get('GraphST_source'):continue
        if any(old['config'].get(k)!=manifest['config'].get(k) for k in (*COMMON,*keys)):continue
        relevant=['adapters.py']
        if method not in ('graphst','cellcharter'):relevant+=['protocol.py',method+'_adapter.py']
        if method=='spaniche':relevant+=['spaniche_runner.R']
        relevant+=[k for k in old['implementation'] if k.startswith('vendor/'+method+'/') or (method=='spaniche' and k.startswith('vendor/intnmf/'))]
        if any(old['implementation'].get(k)!=manifest['implementation'].get(k) for k in relevant):continue
        shutil.copytree(source,target)
        audit={'source':str(source),'unchanged_inputs':True,'matching_method_configuration':True,
               'matching_method_implementation':True,'source_complete_sha256':checksum(source/'complete.json'),
               'embedding_sha256':checksum(source/'embedding.npy'),'labels_sha256':checksum(source/'labels.npy')}
        (target/'reused_from.json').write_text(json.dumps(audit,indent=2)+'\n')
        reused.append(method)
    return reused
