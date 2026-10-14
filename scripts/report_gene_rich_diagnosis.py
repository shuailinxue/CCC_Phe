"""Actual diagnostic maps; infer saved sampled checkpoints, never train.

Figure contract: generative assumptions and supported supervision matter;
network depth alone did not restore the weak planted MI. The controlled
expression dataset is different, so this is not a same-data model benchmark.
Python quantitative grid, 183-mm width, raw cluster colors, editable PDF/SVG.
"""
from pathlib import Path
import sys,json
import numpy as np,pandas as pd,torch,anndata as ad,scipy.sparse as sp
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from phenoniche.storage import STORAGE_ROOT
from phenoniche.gnn.data import GraphData,Targets
from phenoniche.gnn.config import GNNConfig
from phenoniche.gnn.model import model_from_config
from phenoniche.gnn.training import infer
from phenoniche.gnn.reporting import style,COLORS
from phenoniche.simulation.baselines import dictionary_program_projection
from phenoniche.simulation.evaluation import partition_scores,label_mapping,mapped
from sklearn.cluster import MiniBatchKMeans


def main():
    torch.set_num_threads(6);torch.cuda.set_device(2)
    source=STORAGE_ROOT/'outputs/tumor_edge_gene_rich_simulation';base=STORAGE_ROOT/'outputs/tumor_edge_gene_rich_diagnostics'
    control=base/'program_coherent_consistent_mask';folder=control/'standard';cache=control/'input'
    p=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False);pre=np.load(cache/'preprocessing.npz');a=ad.read_h5ad(control/'injected.h5ad');features=pd.DataFrame(p['features']);categories=tuple(p['categories']);types=pd.Categorical(a.obs.cell_type,categories=categories).codes
    rows=[np.asarray(r) for r in p['sample_rows']]
    data=GraphData(sp.csr_matrix(a.X),np.asarray(a.X).sum(1),tuple(a.var_names),p['encoder_indices'],p['gene_mean'],p['gene_scale'],types,categories,np.asarray(a.obsm['spatial']),np.load(cache/'members.npy'),np.load(cache/'anchor_weights.npy'),Targets(np.load(cache/'ccc.npy',mmap_mode='r'),np.arange(len(features))),p['feature_scale'],p['feature_mask'],features,a.obs[['cell_id','cell_type']].reset_index(drop=True),rows,p['folds'],pre['blocks'],np.load(cache/'composition_target.npy'),p['data_audit'],expression_normalization=p['expression_normalization'],library_mean=p['library_mean'],library_scale=p['library_scale'])
    cfg=GNNConfig(**p['config']);model=model_from_config(data,cfg).to('cuda:2');model.load_state_dict(p['model_state']);model.eval()
    sample=np.concatenate(rows);data.prepare_subset_device(sample,'cuda:2')
    with torch.no_grad():
        z_check=np.concatenate([data.forward(model,data.batch(sample[start:start+128],'cuda:2'),reconstruct=False)['Z'].cpu().numpy() for start in range(0,len(sample),128)])
    np.testing.assert_allclose(z_check,np.load(folder/'sample_Z.npy'),rtol=2e-5,atol=2e-5)
    del data._resident;torch.cuda.empty_cache()
    output=folder/'full_inference';output.mkdir(exist_ok=True)
    from phenoniche.simulation.response_simulation import checksum
    identity={'checkpoint':checksum(folder/'checkpoint.pt'),'input':checksum(control/'injected.h5ad')}
    complete=output/'complete.json'
    if complete.exists() and json.loads(complete.read_text())==identity and all((output/(v+'.npy')).exists() for v in ['Z','P','activity','H','B']):
        z=np.load(output/'Z.npy');h=np.load(output/'H.npy')
    else:
        data.prepare_device('cuda:2',output/'cache');arrays=infer(model,data,output,128);z=np.asarray(arrays['Z']);h=np.load(output/'H.npy');complete.write_text(json.dumps(identity,indent=2))
    projection=dictionary_program_projection(h,data.feature_scale)
    clustering=MiniBatchKMeans(n_clusters=4,random_state=40700,n_init=10,batch_size=1024).fit(z[rows[0]]@projection);labels=clustering.predict(z@projection)
    sampled=np.load(folder/'raw_labels.npy');np.testing.assert_array_equal(labels[sample],sampled)
    pd.DataFrame({'cell_id':a.obs.cell_id.to_numpy(),'raw_cluster':labels}).to_csv(output/'raw_clusters.csv',index=False)
    truth=a.obs.true_niche.to_numpy();train=rows[0];etr=train[np.isin(truth[train],[2,3])];mapping=label_mapping(labels,truth,train);ab=label_mapping(labels,truth,etr)
    records=[]
    for name,r in [('sampled_test',rows[2]),('complete_spatial_test',np.flatnonzero(data.folds==2)),('all_transductive',np.arange(len(a)))]:
        records.append({'scope':name,**partition_scores(truth,labels,mapped(labels,mapping),r,ab,etr)})
    pd.DataFrame(records).to_csv(output/'metrics.csv',index=False)
    lr=pd.read_csv(control/'lr.csv');g=a.var_names.get_indexer(list(set(lr.ligand)|set(lr.receptor)));mi=a.var_names.get_indexer(list(set(lr.loc[lr.mi.isin([1,2]),'ligand'])|set(lr.loc[lr.mi.isin([1,2]),'receptor'])));edge=np.isin(truth,[2,3]);raw=np.asarray(a.X)
    audit={'cells':len(a),'input_genes':a.n_vars,'LR_pairs':len(lr),'valid_features':int(data.feature_mask.sum()),'edge_MI_gene_fraction':len(mi)/a.n_vars,'edge_MI_count_fraction_in_edge':float(raw[edge][:,mi].sum()/raw[edge].sum()),'all_LR_count_fraction':float(raw[:,g].sum()/raw.sum()),'checkpoint_reload_reproduces_sample_Z':True,'full_inference_clusters_reproduce_sample_clusters':True,'all_cell_maps_are_transductive':True,'comparison_uses_different_LR_datasets':True,'joint_expression_stage_evaluated':False}
    (output/'audit.json').write_text(json.dumps(audit,indent=2))
    original=ad.read_h5ad(source/'injected.h5ad');non_lr=np.flatnonzero(~original.var_names.isin(set(lr.ligand)|set(lr.receptor)));np.testing.assert_array_equal(original.X[:,non_lr],a.X[:,non_lr]);np.testing.assert_array_equal(original.obsm['spatial'],a.obsm['spatial'])
    old_labels=pd.read_csv(source/'raw_clusters.csv')['CCC_Phe GNN (fixed K)'].to_numpy();xy=data.coordinates
    from matplotlib import pyplot as plt
    style();plt.rcParams.update({'pdf.fonttype':42,'svg.fonttype':'none','font.size':7})
    fig=plt.figure(figsize=(183/25.4,125/25.4),layout='constrained');grid=fig.add_gridspec(2,3,height_ratios=[1.25,1]);maps=[fig.add_subplot(grid[0,k]) for k in range(3)];titles=['a  Ground truth','b  Original data · CCC-only GNN','c  Controlled data · same GNN']
    for ax,values,title in zip(maps,[truth,old_labels,labels],titles):
        ax.scatter(*xy.T,c=np.asarray(COLORS)[values],s=.7,lw=0,rasterized=True);ax.set_title(title,fontsize=7,loc='left');ax.set_aspect('equal');ax.set(xlabel='x (µm)',ylabel='y (µm)',xlim=(-630,630),ylim=(-630,630),xticks=[-600,0,600],yticks=[-600,0,600]);ax.plot([-550,-450],[-550,-550],c='k',lw=1);ax.text(-500,-515,'100 µm',ha='center',fontsize=5)
    for ax,names in zip(maps,[['Core','Exterior','Edge A','Edge B']]+[[f'Raw cluster {j}' for j in range(4)]]*2):
        ax.legend(handles=[plt.Line2D([],[],color=COLORS[j],marker='o',ls='',ms=2,label=v) for j,v in enumerate(names)],loc='upper center',bbox_to_anchor=(.5,-.22),ncol=2,fontsize=5)
    for k,(directory,title) in enumerate([(source/'gnn','d  Original training'),(folder,'e  Controlled training')]):
        ax=fig.add_subplot(grid[1,k]);history=pd.read_csv(directory/'training.csv')
        for split,color in [('train',COLORS[0]),('validation',COLORS[1])]:ax.plot(history.epoch,history[split+'_ccc'],label=split,color=color,lw=1)
        selected=json.loads((directory/'summary.json').read_text())['best_epoch'] if (directory/'summary.json').exists() else json.loads((directory/'heldout_metrics.json').read_text())['best_epoch']
        ax.axvline(selected,c='#777777',ls=':',lw=.7)
        ax.set(title=title,xlabel='Epoch',ylabel='Scaled CCC Huber');ax.legend(fontsize=6)
    text=fig.add_subplot(grid[1,2]);text.axis('off');text.text(0,.96,'Controlled experiment',weight='bold',va='top');text.text(0,.83,'3,000 genes · 75 LR pairs\n32 programs · hidden 64 · one layer\nTrain / validation / test: 1,600 / 600 / 600\nFixed K=4; fitted on training centers\n75 supported CCC features (10% coverage)\nMI genes: 0.8% of panel\nEdge MI counts: '+f'{audit["edge_MI_count_fraction_in_edge"]:.2%}'+'\n\nDifferent LR rate assumptions and mask;\noriginal non-LR counts and geometry retained.\nRaw cluster IDs are independent across maps.\nExploratory, one seed; no SOTA claim.',va='top',fontsize=6,linespacing=1.4)
    figures=base/'figures';figures.mkdir(exist_ok=True)
    fig.canvas.draw()
    for suffix in ['png','pdf','svg']:fig.savefig(figures/('diagnosis_comparison.'+suffix),dpi=450,bbox_inches='tight')
    plt.close(fig)
    pd.DataFrame({'cell_id':a.obs.cell_id.to_numpy(),'x_um':xy[:,0],'y_um':xy[:,1],'true_region':truth,'original_raw_cluster':old_labels,'controlled_raw_cluster':labels}).to_csv(figures/'diagnosis_spatial_source.csv',index=False)
    print(pd.DataFrame(records).to_string(index=False));print(json.dumps(audit,indent=2))

if __name__=='__main__':main()
