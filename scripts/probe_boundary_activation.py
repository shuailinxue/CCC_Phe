"""Frozen-model causal simulation control, not a retrained baseline benchmark.

Keep anatomical labels and non-LR counts. Give core/exterior programs a
an explicit activity floor inside their own regions; retain all graph/CCC
definitions. Inject independent Poisson expression increments before CCC.
"""
from pathlib import Path
import argparse,copy,json,sys
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from diagnose_gene_rich_niches import read_data
from phenoniche.storage import STORAGE_ROOT
from phenoniche.ccc.pairwise import _side_expression,aggregate_pairwise_ccc
from phenoniche.ccc.lr_atlas import LRInteraction,LRAtlas
from phenoniche.gnn.data import Targets
from phenoniche.gnn.model import model_from_config
from phenoniche.gnn.config import GNNConfig
from phenoniche.gnn.training import infer
from phenoniche.gnn.spatial_refinement import spatial_cluster_features
from phenoniche.simulation.config import SimulationConfig
from phenoniche.simulation.programs import program_specs,spatial_activity
from phenoniche.simulation.evaluation import label_mapping,mapped,partition_scores
from phenoniche.simulation.response_simulation import checksum


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--floor',type=float,default=.5);args=parser.parse_args()
    if not 0<args.floor<=1:raise ValueError('Activity floor must be in (0,1]')
    torch.set_num_threads(4);torch.cuda.set_device(2)
    source=STORAGE_ROOT/'outputs/tumor_edge_gene_rich_coherent_simulation'
    current=STORAGE_ROOT/'outputs/tumor_edge_gene_rich_spatial_niches'
    tag=f'{args.floor:g}'.replace('.','')
    out=STORAGE_ROOT/('outputs/tumor_edge_boundary_diagnosis/activity_floor'+tag);out.mkdir(exist_ok=True)
    hashes={name:checksum(source/name) for name in ('injected.h5ad','gnn/checkpoint.pt','gnn/Z.npy','gnn/H.npy','raw_clusters.csv')}
    a,d,payload=read_data(source);base=SimulationConfig.from_json(source/'config.json')
    lr=pd.read_csv(source/'lr.csv');counts=np.asarray(a.X).copy();original=counts.copy()
    act=spatial_activity(a,base);new=act.copy();radius=a.obs.radius_um.to_numpy();floor=args.floor
    core=radius<base.core_radius_um;normal=radius>=base.core_radius_um+base.edge_width_um
    new[core,2]=floor+(1-floor)*act[core,2];new[normal,3]=floor+(1-floor)*act[normal,3]
    rng=np.random.default_rng(base.seed+6831);genes={g:i for i,g in enumerate(a.var_names)};library=a.obs.synthetic_library_factor.to_numpy()
    states=a.obsm['synthetic_MI_cell_state'];changed_genes=[]
    for spec in program_specs(base)[2:]:
        j=spec['MI']-1;links=lr[lr.mi==spec['MI']]
        for side,role in [('ligand',spec['sender']),('receptor',spec['receiver'])]:
            cells=np.flatnonzero(a.obs.cell_type.to_numpy()==role)
            for gene,loading in zip(links[side],links.loading):
                g=genes[gene];delta=3.*states[cells,j]*library[cells]*(new[cells,j]-act[cells,j])*loading/max(links.loading)
                counts[cells,g]+=rng.poisson(delta);changed_genes.append(g)
    unaffected=np.setdiff1d(np.arange(a.n_vars),changed_genes)
    np.testing.assert_array_equal(counts[:,unaffected],original[:,unaffected])
    assert (counts>=original).all() and np.equal(counts,np.floor(counts)).all()
    a.X=counts;a.obsm['designed_MI_activity']=new;a.obsm['injected_sender_activity']=new.copy();a.obsm['injected_receiver_activity']=new.copy()
    a.write_h5ad(out/'injected.h5ad')
    cache=out/'input';cache.mkdir(exist_ok=True)
    atlas=LRAtlas(tuple(LRInteraction(str(v.lr_id),v.ligand,v.receptor,(v.ligand,),(v.receptor,)) for v in lr.itertuples()),{},'Same real LR subset')
    ligand,receptor=_side_expression(counts,tuple(a.var_names),atlas)
    c=np.lib.format.open_memmap(cache/'ccc.npy',mode='w+',dtype=np.float32,shape=d.targets.matrix.shape)
    print('Recompute real CCC from modified expression; original KNN30/sigma20/opportunity normalization',flush=True)
    aggregate_pairwise_ccc(d.coordinates,d.types,ligand,receptor,5,20.,d.members,d.weights,batch_size=128,lr_batch_size=16,device=torch.device('cuda:2'),communication_out=c)
    c.flush();altered=copy.copy(d);altered.expression=sp.csr_matrix(counts);altered.totals=counts.sum(1);altered.targets=Targets(c,np.arange(c.shape[1]))
    cfg=GNNConfig(**payload['config']);model=model_from_config(altered,cfg).to('cuda:2');model.load_state_dict(payload['model_state']);model.eval()
    altered.prepare_device('cuda:2',out/'cache')
    values=infer(model,altered,out/'gnn',128)
    classifier=torch.load(current/'gnn/niche_model.pt',map_location='cpu',weights_only=False)
    x=np.asarray(values['Z'])@classifier['program_projection']
    x=spatial_cluster_features(x,np.asarray(d.members),np.asarray(d.weights),d.rows,.2)
    pred=classifier['model'].predict(x) # no re-fitting on the altered data
    # Same anatomical labels and train-based evaluation alignment.
    truth=a.obs.true_niche.to_numpy();train,val,test=d.rows;edge=np.isin(truth,[2,3]);etr=train[edge[train]]
    alignment=mapped(pred,label_mapping(pred,truth,train));ab=label_mapping(pred,truth,etr);records=[]
    for scope,r in [('validation',val),('heldout_test',test),('whole_tissue_descriptive',np.arange(len(a)))]:
        records.append({'scope':scope,'core_errors':int(((truth[r]==0)&(alignment[r]!=0)).sum()),
            'exterior_errors':int(((truth[r]==1)&(alignment[r]!=1)).sum()),**partition_scores(truth,pred,alignment,r,ab,etr)})
    pd.DataFrame(records).to_csv(out/'metrics.csv',index=False);pd.DataFrame({'cell_id':a.obs.cell_id,'raw_cluster':pred}).to_csv(out/'raw_clusters.csv',index=False)
    for name,digest in hashes.items():assert checksum(source/name)==digest,name
    (out/'summary.json').write_text(json.dumps({'activity_floor':floor,'changed_genes':len(set(changed_genes)),
        'added_RNA_count_fraction':float((counts-original).sum()/original.sum()),'anatomical_labels_unchanged':True,
        'non_LR_expression_unchanged':True,'GNN_and_clustering_frozen':True,'CCC_recomputed_from_actual_expression':True,
        'baseline_ranking_valid':False,'scope':'Frozen-model counterfactual on changed data. All methods need retraining/re-evaluation before a fair benchmark.',
        'source_sha256':hashes,'seed':base.seed+6831},indent=2)+'\n')
    print(pd.DataFrame(records).to_string(index=False),flush=True)


if __name__=='__main__':main()
