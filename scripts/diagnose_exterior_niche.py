"""Read-only exterior error diagnosis; no fitting/updating of the GNN.

Fit clustering on existing training Z only. True regions are evaluation-only.
Local consensus is an explicitly separate exploratory postprocessing step;
held-out receivers cannot use donor neighborhoods from another physical fold.
"""
from pathlib import Path
import argparse,copy,json,sys
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.cluster import KMeans,MiniBatchKMeans
from sklearn.metrics import silhouette_score

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from diagnose_gene_rich_niches import read_data
from phenoniche.storage import STORAGE_ROOT
from phenoniche.gnn.model import model_from_config
from phenoniche.gnn.config import GNNConfig
from phenoniche.simulation.baselines import dictionary_program_projection
from phenoniche.simulation.evaluation import label_mapping,mapped,partition_scores
from phenoniche.simulation.response_simulation import checksum


def local_weights(members,weights,rows):
    owner=np.full(len(members),-1,np.int8)
    for fold,r in enumerate(rows):
        cells=np.unique(members[r])
        if (owner[cells]>=0).any():raise ValueError('Physical folds overlap')
        owner[cells]=fold
    donor_owner=owner[members]
    pure=np.where((donor_owner==donor_owner[:,:1]).all(1),donor_owner[:,0],-1)
    valid=(members!=np.arange(len(members))[:,None])
    valid &= (owner[:,None]<0)|((pure[members]==owner[:,None])&(pure[members]>=0))
    return np.asarray(weights)*valid,valid


def consensus(labels,members,weights,rows,agreement=.8):
    w,valid=local_weights(members,weights,rows)
    vote=np.stack([(w*(labels[members]==k)).sum(1) for k in range(4)],1)
    mass=vote.sum(1);winner=vote.argmax(1)
    confidence=vote.max(1)/np.maximum(mass,1e-12)
    update=(confidence>=agreement)&(valid.sum(1)>=5)
    return np.where(update,winner,labels).astype(np.int16)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,default=STORAGE_ROOT/'outputs/tumor_edge_gene_rich_coherent_simulation')
    parser.add_argument('--output',type=Path,default=STORAGE_ROOT/'outputs/tumor_edge_exterior_diagnosis')
    args=parser.parse_args();source=args.source;out=args.output
    if out.is_symlink() or not out.resolve().is_relative_to(STORAGE_ROOT.resolve()) or out.resolve()==source.resolve():raise ValueError('Independent direct NAS output required')
    out.mkdir(exist_ok=True,parents=True)
    names=('injected.h5ad','gnn/checkpoint.pt','gnn/Z.npy','gnn/H.npy','raw_clusters.csv','input/ccc.npy')
    hashes={k:checksum(source/k) for k in names}
    torch.set_num_threads(4);torch.cuda.set_device(2)
    a,data,payload=read_data(source);train,val,test=data.rows
    z=np.load(source/'gnn/Z.npy');h=np.load(source/'gnn/H.npy')
    native=z@dictionary_program_projection(h,data.feature_scale)
    reconstruction=z@h[:,data.feature_mask];scaled=reconstruction/data.feature_scale[data.feature_mask]
    intensity=np.log1p(z.sum(1));reference=np.median(intensity[train]);spread=max(float(intensity[train].std()),.1)
    relative=np.column_stack((np.sqrt(reconstruction/np.maximum(reconstruction.sum(1,keepdims=True),1e-8)),.25*(intensity-reference)/spread))
    inputs={'Native full KMeans':native,'Log CCC KMeans':np.log1p(scaled),'Relative CCC + intensity':relative}
    raw=pd.read_csv(source/'raw_clusters.csv')['CCC_Phe GNN (fixed K)'].to_numpy()
    mb=MiniBatchKMeans(n_clusters=4,n_init=10,random_state=40700,batch_size=min(1024,len(train))).fit(native[train])
    np.testing.assert_array_equal(raw,mb.predict(native))
    candidates={'Original GNN':raw};representations={'Original GNN':native}
    for name,x in inputs.items():
        candidates[name]=KMeans(n_clusters=4,n_init=10,random_state=40700).fit(x[train]).predict(x)
        representations[name]=x
    # Predeclared 20% local averaging of the clustering representation only.
    # Z, H, measured C and the GNN checkpoint stay exactly unchanged.
    w,valid=local_weights(np.asarray(data.members),np.asarray(data.weights),data.rows)
    neighbor=(native[np.asarray(data.members)]*w[:,:,None]).sum(1)/np.maximum(w.sum(1,keepdims=True),1e-12)
    averaged=np.where((valid.sum(1)>=5)[:,None],.8*native+.2*neighbor,native)
    candidates['Local 20% + full KMeans']=KMeans(n_clusters=4,n_init=10,random_state=40700).fit(averaged[train]).predict(averaged)
    representations['Local 20% + full KMeans']=averaged
    for name in ('Original GNN','Native full KMeans','Local 20% + full KMeans','Log CCC KMeans'):
        new=name+' + consensus'
        candidates[new]=consensus(candidates[name],np.asarray(data.members),np.asarray(data.weights),data.rows)
        representations[new]=representations[name]
    # Observed C is an oracle control, never an encoder/production input.
    actual=np.asarray(data.targets.matrix)[:,data.feature_mask]/data.feature_scale[data.feature_mask]
    candidates['Observed CCC (diagnostic)']=KMeans(n_clusters=4,n_init=10,random_state=40700).fit(actual[train]).predict(actual)
    representations['Observed CCC (diagnostic)']=actual
    # All transforms, clustering, and consensus above are label-blind.
    truth=a.obs.true_niche.to_numpy();edge=np.isin(truth,[2,3]);normal=truth==1
    cross_edge=np.isin(truth[np.asarray(data.members)],[2,3]).any(1)
    far=normal&~cross_edge;records=[]
    for name,pred in candidates.items():
        mapping=label_mapping(pred,truth,train);aligned=mapped(pred,mapping);ab=label_mapping(pred,truth,train[edge[train]])
        x=representations[name]
        sil=float(silhouette_score(x[val],pred[val],sample_size=min(1200,len(val)),random_state=40700))
        for scope,r in [('all_centers_descriptive',np.arange(len(a))),('validation',val),('heldout_test',test)]:
            records.append({'method':name,'scope':scope,'validation_silhouette':sil,
                'normal_to_A':int(((truth[r]==1)&(aligned[r]==2)).sum()),
                'far_normal_to_A':int((far[r]&(aligned[r]==2)).sum()),
                'normal_to_any_edge':int((normal[r]&np.isin(aligned[r],[2,3])).sum()),
                'far_normal_to_any_edge':int((far[r]&np.isin(aligned[r],[2,3])).sum()),
                'changed_labels_after_training_alignment':int((aligned[r]!=mapped(raw,label_mapping(raw,truth,train))[r]).sum()),
                **partition_scores(truth,pred,aligned,r,ab,train[edge[train]])})
    pd.DataFrame(records).to_csv(out/'clustering_controls.csv',index=False)
    pd.DataFrame({'cell_id':a.obs.cell_id.to_numpy(),**candidates}).to_csv(out/'candidate_raw_labels.csv',index=False)
    original=mapped(raw,label_mapping(raw,truth,train));false=normal&(original==2);far_false=false&far
    stability=[]
    for name,x in [('Native full KMeans',native),('Local 20% + full KMeans',averaged)]:
        for seed in (40701,40702):
            pred=KMeans(n_clusters=4,n_init=10,random_state=seed).fit(x[train]).predict(x)
            from sklearn.metrics import adjusted_rand_score
            alignment=mapped(pred,label_mapping(pred,truth,train))
            stability.append({'method':name,'seed':seed,'validation_ARI_vs_seed40700':adjusted_rand_score(candidates[name][val],pred[val]),
                'far_normal_to_any_edge':int((far&np.isin(alignment,[2,3])).sum())})
    pd.DataFrame(stability).to_csv(out/'full_kmeans_seed_stability.csv',index=False)
    # Paired forward-only probe. No gradients, no checkpoint selection, no training.
    groups={'far_false_A':np.flatnonzero(far_false),
            'normal_control':np.flatnonzero(normal&(original==1)&far)[:22],
            'edge_A_control':np.flatnonzero(truth==2)[:22]}
    selected=np.concatenate(list(groups.values()));model=model_from_config(data,GNNConfig(**payload['config'])).to('cuda:2')
    model.load_state_dict(payload['model_state']);model.eval()
    true_h=np.load(source/'ground_truth/MI_dictionary.npy');den=np.maximum(true_h.sum(1),1e-12)
    lr=pd.read_csv(source/'lr.csv');links=lr[lr.mi.isin([1,2])]
    genes=a.var_names.get_indexer(sorted(set(links.ligand)|set(links.receptor)))
    altered=copy.copy(data);counts=np.asarray(a.X).copy();counts[:,genes]=0
    altered.expression=sp.csr_matrix(counts);altered.totals=counts.sum(1)
    values={}
    with torch.no_grad():
        for name,d in [('original',data),('edge_LR_zeroed',altered)]:
            values[name]=d.forward(model,d.batch(selected,'cuda:2',include_target=False),reconstruct=False)['Z'].cpu().numpy()
    np.testing.assert_allclose(values['original'],z[selected],rtol=3e-5,atol=3e-5)
    probes=[];start=0
    for group,r in groups.items():
        for name,value in values.items():
            scores=value[start:start+len(r)]@(h@true_h.T)/den
            probes.append({'group':group,'condition':name,'n':len(r),**{f'predicted_MI{j+1}':float(scores[:,j].mean()) for j in range(4)}})
        start+=len(r)
    pd.DataFrame(probes).to_csv(out/'forward_only_LR_probe.csv',index=False)
    # Distance decomposition of far exterior errors using actual learned centroids.
    normal_cluster=next(k for k,v in label_mapping(raw,truth,train).items() if v==1)
    edge_cluster=next(k for k,v in label_mapping(raw,truth,train).items() if v==2)
    dictionary=h[:,data.feature_mask]/data.feature_scale[data.feature_mask]
    native_centers=mb.cluster_centers_
    projection=dictionary_program_projection(h,data.feature_scale)
    centers_z=np.linalg.lstsq(projection.T,native_centers.T,rcond=1e-6)[0].T
    # Pseudoinverse gives an equivalent distance in active CCC feature space.
    centers=centers_z@dictionary
    difference=((scaled[far_false]-centers[normal_cluster])**2-(scaled[far_false]-centers[edge_cluster])**2).mean(0)
    table=data.features.loc[data.feature_mask].copy();table['distance_advantage_for_A']=difference
    table.sort_values('distance_advantage_for_A',ascending=False).to_csv(out/'far_exterior_centroid_distance_features.csv',index=False)
    summary={'cells':len(a),'normal_cells':int(normal.sum()),'normal_to_A':int(false.sum()),
        'normal_to_A_with_true_edge_in_KNN':int((false&cross_edge).sum()),'far_normal_to_A':int(far_false.sum()),
        'model_changed':False,'CCC_formula_changed':False,'notebook_changed':False,
        'training_targets_used_in_clustering':False,'region_labels_used_in_clustering':False,
        'selection':'Exploration only; no automatic choice using true labels. Validation silhouettes and independent error metrics are exported separately.',
        'local_representation':'80% own inferred ZH distance representation + 20% Gaussian-weighted local mean; >=5 eligible donors, physical-fold restrictions identical to consensus. Used for clustering only, not exported as modified Z/H/CCC.',
        'consensus':'One pass; >=80% existing Gaussian neighbor vote, >=5 donors; observed physical-fold ownership restricts held-out donors.',
        'source_sha256':hashes}
    for name,digest in hashes.items():
        if checksum(source/name)!=digest:raise RuntimeError('Original result changed: '+name)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    # Quantitative comparison contract: same cells, weights and raw CCC; four
    # matched spatial panels, same previously misassigned centers highlighted.
    # Color matching reads truth after all clustering, purely for display.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from types import SimpleNamespace
    from phenoniche.simulation.joint_reporting import JointExpressionReport
    plt.rcParams.update({'font.family':'sans-serif','font.size':8,'pdf.fonttype':42,'svg.fonttype':'none','legend.frameon':False})
    context=SimpleNamespace(xy=data.coordinates,show=np.ones(len(a),bool))
    fig,axes=plt.subplots(1,4,figsize=(12,3.7),layout='constrained')
    score=pd.DataFrame(records).query("scope=='all_centers_descriptive'").set_index('method')
    sequence=[('Ground truth',truth),('Original GNN',original),('Native full KMeans',mapped(candidates['Native full KMeans'],label_mapping(candidates['Native full KMeans'],truth,train))),
        ('Local 20% + full KMeans',mapped(candidates['Local 20% + full KMeans'],label_mapping(candidates['Local 20% + full KMeans'],truth,train)))]
    titles=['a  Ground truth','b  Original GNN','c  Full KMeans','d  Local 20% + full KMeans']
    for ax,(name,pred),title in zip(axes,sequence,titles):
        if name!='Ground truth':title+=f'\nFar exterior errors: {int(score.loc[name,"far_normal_to_any_edge"])} · ARI={score.loc[name,"overall_ARI"]:.3f}'
        JointExpressionReport.panel(context,ax,pred,title,True,names=['Tumor core','Normal exterior','Edge A','Edge B'] if name=='Ground truth' else None)
        ax.scatter(*data.coordinates[far_false].T,s=14,facecolors='none',edgecolors='#333333',lw=.55,rasterized=True)
    fig.supxlabel('Same GNN weights, Z, H and CCC · 8,000 cells · circles mark the same 22 original far-exterior errors\nRegion labels used only for evaluation/color matching; pooling coefficient fixed before evaluating results. Exploratory postprocessing, not a new baseline ranking.',fontsize=7)
    folder=out/'figures';folder.mkdir(exist_ok=True)
    for ext in ('png','pdf','svg'):fig.savefig(folder/f'exterior_error_comparison.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)
    print(pd.DataFrame(records).query("scope=='all_centers_descriptive'").to_string(index=False),flush=True)
    print(pd.DataFrame(probes).to_string(index=False),flush=True)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
