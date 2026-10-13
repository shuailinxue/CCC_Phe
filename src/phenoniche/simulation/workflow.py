"""Restartable simulated benchmark. Full execution is invoked only by user Run All."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd,anndata as ad
from ..storage import STORAGE_ROOT
from ..gnn.training import fit,infer
from ..gnn.pipeline import load_checkpoint
from ..gnn.evaluation import discover
from .generation import generate,check_environment
from .injection import inject
from .adapter import adapt,gnn_config
from .baselines import run_baselines,fixed_clustering
from .evaluation import evaluate
from .config import SimulationConfig


def protect(config):
    root=Path(config.output_dir)
    if root.is_symlink() or not root.resolve().is_relative_to(STORAGE_ROOT.resolve()):raise ValueError('Simulation outputs require a direct NAS workspace path')
    for p in ('outputs/xenium_prime_5k','outputs/microenvironment_gnn','outputs/microenvironment_gnn_expr05','outputs/microenvironment_gnn_composition_expr05','outputs/microenvironment_gnn_composition_weak_expr05','outputs/microenvironment_gnn_ccc_only_expr05','data'):
        old=STORAGE_ROOT/p
        if root.resolve()==old or root.resolve().is_relative_to(old) or old.is_relative_to(root.resolve()):raise ValueError('Simulation output overlaps protected real-data results')
    root.mkdir(parents=True,exist_ok=True);path=root/'config.json'
    if path.exists() and json.loads(path.read_text())!=config.to_dict():raise ValueError('Simulation config changed; use an independent output directory rather than overwrite checkpoints')
    path.write_text(json.dumps(config.to_dict(),indent=2)+'\n')
    return root


def dataset(config):
    output=protect(config);versions=check_environment()
    before=output/'before_injection.h5ad';after=output/'injected.h5ad'
    if before.exists() and (output/'lr.csv').exists():base=ad.read_h5ad(before);lr=pd.read_csv(output/'lr.csv')
    else:base,lr=generate(config,output)
    if after.exists():a=ad.read_h5ad(after)
    else:
        a=inject(base,lr,config);a.write_h5ad(output/'injected.incomplete.h5ad');(output/'injected.incomplete.h5ad').replace(after)
    if len(a)!=config.n_cells or not np.array_equal(a.obs.cell_id,base.obs.cell_id):raise ValueError('Cached simulation identity/shape changed')
    gt=output/'ground_truth';gt.mkdir(exist_ok=True)
    np.save(gt/'niche_labels_before.npy',base.obs.true_niche.to_numpy());np.save(gt/'MI_activity_before.npy',np.zeros((len(base),2),np.float32))
    (output/'environment.json').write_text(json.dumps(versions,indent=2))
    return a,lr


def run_experiment(config):
    output=protect(config)
    required=('metrics.csv','diagnostics.json','MI_matches.csv','raw_clusters.csv','radial_composition_audit.csv','gnn/Z.npy','gnn/H.npy','gnn/B.npy','gnn/K_selection.csv','ground_truth/MI_activity.npy','complete.json')
    evaluation_signature=hashlib.sha256((Path(__file__).parent/'evaluation.py').read_bytes()).hexdigest()
    if all((output/p).is_file() for p in required) and json.loads((output/'complete.json').read_text()).get('evaluation_signature')==evaluation_signature:return output
    adata,lr=dataset(config);data=adapt(adata,lr,config,output)
    np.save(output/'ground_truth/composition_before.npy',data.composition_target)
    gcfg=gnn_config(config,output/'gnn');Path(gcfg.output_dir).mkdir(exist_ok=True)
    if (Path(gcfg.output_dir)/'checkpoint.pt').exists():model,_=load_checkpoint(data,gcfg)
    else:model=fit(data,gcfg,initialize_composition=config.initialize_composition,rescale_activity=config.rescale_activity)
    arrays=infer(model,data,gcfg.output_dir,gcfg.batch_size);h=np.load(Path(gcfg.output_dir)/'H.npy');b=np.load(Path(gcfg.output_dir)/'B.npy')
    k,adaptive,_=discover(data,arrays,h,gcfg)
    labels,ae=run_baselines(data,config,output/'baselines')
    latent=np.column_stack((np.sqrt(np.asarray(arrays['P'])),.25*np.log1p(np.asarray(arrays['activity']))))
    labels['CCC_Phe GNN (fixed K)']=fixed_clustering(latent,data.rows[0],config)
    labels['CCC_Phe GNN (selected K)']=adaptive
    pd.DataFrame({'cell_id':adata.obs.cell_id.to_numpy(),**labels}).to_csv(output/'raw_clusters.csv',index=False)
    evaluate(adata,data,labels,arrays,h,b,ae,output)
    (output/'complete.json').write_text(json.dumps({'simulation_signature':config.signature(),'evaluation_signature':evaluation_signature,'GNN_selected_K':k,'truth_used_in_training_or_K_selection':False,'raw_labels_are_unaligned':True,'scope':'synthetic independent downstream only'},indent=2)+'\n')
    return output


def run_from_config(path):return run_experiment(SimulationConfig.from_json(path))
