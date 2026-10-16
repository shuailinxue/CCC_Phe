"""Condition-blind resolution/metric probes of a completed GNN; no refitting."""
from pathlib import Path
import sys,json,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,pandas as pd,torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score,adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
import igraph as ig
import leidenalg
from phenoniche.gnn.communication_metric import communication_embedding
from phenoniche.gnn.spatial_refinement import spatial_cluster_features


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    torch.set_num_threads(8);s=args.source;o=args.output;o.mkdir(parents=True,exist_ok=True)
    frame=pd.read_csv(s/'assignments.csv.gz');p=np.load(s/'gnn/P.npy');a=np.load(s/'gnn/activity.npy');h=np.load(s/'gnn/H.npy');pre=np.load(s/'input/preprocessing.npz')
    m=np.load(s/'input/members.npy');w=np.load(s/'input/weights.npy');n=len(p);rng=np.random.default_rng(52173);fit=np.sort(rng.choice(n,min(n,12000),replace=False));sample=np.sort(rng.choice(n,min(n,1500),replace=False))
    target=np.load(s/'input/ccc.npy',mmap_mode='r');probe=np.sort(rng.choice(n,min(n,4096),replace=False))
    rms=np.sqrt(np.mean(np.asarray(target[probe],dtype=np.float64)**2,axis=0)).astype(np.float32);rms=np.maximum(rms,1e-6)
    representations={}
    for name,scale in [('current_partial_RMS',pre['feature_scale']),('full_feature_RMS',rms)]:
        values,_,_=communication_embedding(p,a,h,scale,fit)
        representations[name]=spatial_cluster_features(values,m,w,[np.arange(n),np.empty(0,int),np.empty(0,int)],.2)
    records=[];counts=[]
    def save(name,metric,method,k,labels,sil,stability):
        freq=pd.crosstab(frame.condition,labels,normalize='index').reindex(index=['HC','DM'],columns=np.unique(labels),fill_value=0)
        rare=(freq.loc['HC']<=.001)&(freq.loc['DM']>=.01)
        records.append({'name':name,'metric':metric,'method':method,'K':k,'silhouette':sil,'stability':stability,'shared_mass':np.minimum(freq.loc['HC'],freq.loc['DM']).sum(),'case_rare_niches':int(rare.sum()),'DM_fraction_in_rare_niches':freq.loc['DM',rare].sum()})
        for col in freq:counts.append({'name':name,'niche':int(col)+1,'HC_fraction':freq.loc['HC',col],'DM_fraction':freq.loc['DM',col]})
        np.save(o/(name+'_labels.npy'),labels)
    for metric,x in representations.items():
        np.save(o/(metric+'_embedding.npy'),x)
        for k in [6,10,14,18,24]:
            km=KMeans(n_clusters=k,n_init=10,random_state=52173).fit(x[fit]);other=KMeans(n_clusters=k,n_init=10,random_state=52174).fit(x[fit]);label=km.predict(x)
            save(metric+'_k'+str(k),metric,'KMeans',k,label,float(silhouette_score(x[sample],label[sample])),float(adjusted_rand_score(label[sample],other.predict(x[sample]))))
        nn=NearestNeighbors(n_neighbors=21,n_jobs=8).fit(x);distance,index=nn.kneighbors(x)
        sigma=np.maximum(distance[:,-1],1e-8);row=np.repeat(np.arange(n),20);col=index[:,1:].ravel();dist=distance[:,1:].ravel()
        weights=np.exp(-dist**2/np.maximum(sigma[row]*sigma[col],1e-8));g=ig.Graph(n=n,edges=np.column_stack([row,col]),directed=True);g.es['weight']=weights;g=g.as_undirected(combine_edges='mean')
        for res in [.3,.6,1.,1.5]:
            part=leidenalg.find_partition(g,leidenalg.RBConfigurationVertexPartition,weights='weight',resolution_parameter=res,seed=52173)
            second=leidenalg.find_partition(g,leidenalg.RBConfigurationVertexPartition,weights='weight',resolution_parameter=res,seed=52174)
            label=np.asarray(part.membership);k=len(set(label))
            save(metric+'_leiden'+str(res),metric,'Leiden',k,label,float(silhouette_score(x[sample],label[sample])),float(adjusted_rand_score(label,second.membership)))
        pd.DataFrame(records).to_csv(o/'metric_resolution_probes.csv',index=False);pd.DataFrame(counts).to_csv(o/'all_niche_fractions.csv',index=False)
        print(pd.DataFrame(records).tail(9).to_string(index=False),flush=True)
    (o/'audit.json').write_text(json.dumps({'training':False,'condition_in_clustering':False,'condition_in_metrics':True,'note':'All probes retained; no automatic choice based on disease exclusivity','RMS_fit_rows':probe.tolist()},indent=2))

if __name__=='__main__':main()
