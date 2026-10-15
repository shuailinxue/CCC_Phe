"""Limited label-blind clustering controls and independent truth-definition audit.

No changes to data, model, CCC formula or existing Notebook. Anatomical and
injected physical-program truth are reported separately, without relabeling.
"""
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score,adjusted_rand_score

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from diagnose_gene_rich_niches import read_data
from phenoniche.storage import STORAGE_ROOT
from phenoniche.gnn.spatial_refinement import spatial_cluster_features
from phenoniche.simulation.baselines import dictionary_program_projection
from phenoniche.simulation.evaluation import label_mapping,mapped,partition_scores
from phenoniche.simulation.response_simulation import checksum


def main():
    source=STORAGE_ROOT/'outputs/tumor_edge_gene_rich_coherent_simulation'
    current=STORAGE_ROOT/'outputs/tumor_edge_gene_rich_spatial_niches'
    out=STORAGE_ROOT/'outputs/tumor_edge_boundary_diagnosis';out.mkdir(exist_ok=True)
    hashes={name:checksum(source/name) for name in ('injected.h5ad','gnn/checkpoint.pt','gnn/Z.npy','gnn/H.npy','raw_clusters.csv')}
    a,d,payload=read_data(source);train,val,test=d.rows
    z=np.load(source/'gnn/Z.npy');h=np.load(source/'gnn/H.npy')
    # Only original training CCC fits the feature normalization alternatives.
    rms=np.maximum(np.sqrt(np.square(d.targets[train].astype(float)).mean(0)),1e-6)
    global_scale=max(float(np.median(rms[d.feature_mask])),1e-6)
    candidates={};features={};fit_details={}
    for power,mixing in [(.5,.1),(.5,.2),(.5,.3),(.0,.2),(.25,.2),(.75,.2),(1.,.2)]:
        scale=global_scale*(rms/global_scale)**power
        x=z@dictionary_program_projection(h,scale)
        x=spatial_cluster_features(x,np.asarray(d.members),np.asarray(d.weights),d.rows,mixing)
        name=f'KMeans scale={power:g} local={mixing:g}'
        model=KMeans(n_clusters=4,n_init=10,random_state=40700).fit(x[train])
        candidates[name]=model.predict(x);features[name]=x
        fit_details[name]={'solver':'KMeans','scale_power':power,'mixing':mixing}
    base=features['KMeans scale=0.5 local=0.2']
    # PCA preserves 99.5% training variance and avoids singular dictionary axes.
    transform=PCA(n_components=.995,svd_solver='full').fit(base[train]);x=transform.transform(base)
    start=KMeans(n_clusters=4,n_init=10,random_state=40700).fit(x[train])
    covariance_floor=max(1e-6,float(x[train].var(0).mean())*1e-3)
    for covariance in ('spherical','diag','tied','full'):
        name='GMM '+covariance
        model=GaussianMixture(n_components=4,covariance_type=covariance,reg_covar=covariance_floor,
            n_init=1,max_iter=300,random_state=40700,means_init=start.cluster_centers_).fit(x[train])
        candidates[name]=model.predict(x);features[name]=x
        fit_details[name]={'solver':'GMM','covariance':covariance,'PCA_components':int(transform.n_components_),
            'converged':bool(model.converged_),'weights':model.weights_.tolist(),'validation_log_likelihood':float(model.score(x[val]))}
    # All candidates above fit without the true region/MI labels.
    truth=a.obs.true_niche.to_numpy();edge=np.isin(truth,[2,3]);far=(truth==1)&~edge[np.asarray(d.members)].any(1)
    records=[];tr_edge=train[edge[train]]
    for name,pred in candidates.items():
        alignment=mapped(pred,label_mapping(pred,truth,train));ab=label_mapping(pred,truth,tr_edge)
        sil=float(silhouette_score(features[name][val],pred[val],sample_size=min(1200,len(val)),random_state=40700))
        for scope,r in [('validation',val),('heldout_test',test),('whole_tissue_descriptive',np.arange(len(a)))]:
            records.append({'method':name,'scope':scope,'validation_silhouette':sil,
                'core_errors':int(((truth[r]==0)&(alignment[r]!=0)).sum()),
                'exterior_errors':int(((truth[r]==1)&(alignment[r]!=1)).sum()),
                'far_exterior_errors':int((far[r]&np.isin(alignment[r],[2,3])).sum()),
                **partition_scores(truth,pred,alignment,r,ab,tr_edge)})
    pd.DataFrame(records).to_csv(out/'clustering_controls.csv',index=False)
    pd.DataFrame({'cell_id':a.obs.cell_id.to_numpy(),**candidates}).to_csv(out/'raw_candidate_labels.csv',index=False)
    (out/'fit_details.json').write_text(json.dumps(fit_details,indent=2)+'\n')
    # Known injected activities from physical pairs, never observed CCC or Z.
    known=np.load(source/'ground_truth/MI_activity.npy')
    order=np.array([2,3,0,1]);dominant=order[known.argmax(1)]
    defined=known.max(1)>1e-8
    pd.DataFrame({'cell_id':a.obs.cell_id.to_numpy(),'anatomical_truth':truth,
        'known_physical_program_dominance':np.where(defined,dominant,-1),
        **{f'known_MI{j+1}':known[:,j] for j in range(4)}}).to_csv(out/'independent_program_truth_audit.csv.gz',index=False)
    saved=pd.read_csv(current/'baseline_comparison/labels.csv')
    definitions=[]
    for method in ('GraphST','CCC-only GNN'):
        pred=saved[method].to_numpy();aligned=mapped(pred,label_mapping(pred,truth,train))
        for scope,r in [('validation',val),('heldout_test',test),('whole_tissue_descriptive',np.arange(len(a)))]:
            rr=r[defined[r]]
            definitions.append({'method':method,'scope':scope,'n':len(rr),
                'anatomical_ARI':adjusted_rand_score(truth[rr],pred[rr]),
                'known_physical_program_ARI':adjusted_rand_score(dominant[rr],pred[rr]),
                'true_definition_disagreements':int((dominant[rr]!=truth[rr]).sum()),
                'anatomical_errors_already_disagreeing_with_program_truth':int(((aligned[rr]!=truth[rr])&(dominant[rr]!=truth[rr])).sum())})
    pd.DataFrame(definitions).to_csv(out/'truth_definition_control.csv',index=False)
    radius=a.obs.radius_um.to_numpy();width=50.
    # Cell-level generative ramps are recorded directly from radius/profile.
    profile=np.where(truth==0,np.clip((250-radius)/width,0,1),np.where(truth==1,np.clip((radius-350)/width,0,1),np.nan))
    native_pred=candidates['KMeans scale=0.5 local=0.2'];aligned=mapped(native_pred,label_mapping(native_pred,truth,train));details=[]
    for region in (0,1):
        for correct in (True,False):
            keep=(truth==region)&((aligned==truth)==correct)
            details.append({'region':region,'correct_anatomical_assignment':correct,'n':int(keep.sum()),
                'mean_own_cell_program_design':float(profile[keep].mean()),
                'mean_distance_to_anatomical_boundary':float(np.abs(radius[keep]-(250 if region==0 else 350)).mean()),
                'physical_program_definition_disagrees':int((dominant[keep]!=truth[keep]).sum())})
    pd.DataFrame(details).to_csv(out/'boundary_activity_ramp_audit.csv',index=False)
    for name,digest in hashes.items():assert checksum(source/name)==digest,name
    (out/'summary.json').write_text(json.dumps({'source':str(source),'input_sha256':hashes,'model_changed':False,'original_labels_changed':False,
        'truth_definition_control':'Diagnostic only: independent planted physical activity argmax, all methods evaluated under both definitions; anatomical labels remain primary.',
        'scope':'Small clustering controls on one saved simulation, no automatically selected replacement.'},indent=2)+'\n')
    print(pd.DataFrame(records).query("scope=='whole_tissue_descriptive'").to_string(index=False),flush=True)
    print(pd.DataFrame(definitions).to_string(index=False),flush=True)
    print(pd.DataFrame(details).to_string(index=False),flush=True)


if __name__=='__main__':main()
