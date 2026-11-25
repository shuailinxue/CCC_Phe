"""Read-only diagnostics of the saved CRC h and observed target geometry.

ROI selection is downstream interpretation, never training or feature selection.
Nearest neighbors are evaluated within declared random subsets, not all cells.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(key,'6')
from pathlib import Path
import json
import numpy as np
import pandas as pd
import anndata as ad
import torch
from scipy.stats import spearmanr

BASE=Path('/home/nas3/biod/xueshuailin/CCC_Phe/workspace/outputs/crc_neutrophil_xenium/joint_five')
FIT=BASE/'fits/f3441df4f7ed4368'
MATRIX=Path(json.loads((FIT/'complete.json').read_text())['matrices'])
OUT=BASE/'tumor_niche_analysis/h_audit'


def run():
    torch.set_num_threads(6);OUT.mkdir(exist_ok=True)
    rng=np.random.default_rng(40700)
    a=ad.read_h5ad('/home/nas3/biod/xueshuailin/CCC_Phe/workspace/data/crc_neutrophil_xenium/processed/CRC_selected_five.h5ad',backed='r')
    typ=a.obs.cell_type.to_numpy();sample=a.obs['sample'].to_numpy()
    cluster=Path(json.loads((BASE/'figures/umap_K12.json').read_text())['clustering'])
    h=np.load(cluster/'joint_embedding.npy',mmap_mode='r');labels=np.load(cluster/'labels.npy')+1
    roi=np.load(BASE/'tumor_niche_analysis/user_box_mask.npy')
    comp=np.load(MATRIX/'composition.npy',mmap_mode='r');ccc=np.load(MATRIX/'ccc.npy',mmap_mode='r')
    mask=np.load(FIT/'feature_mask.npy');scale=np.load(FIT/'feature_scale.npy')[mask]
    state={k:v.numpy() for k,v in torch.load(FIT/'gnn/joint_neighborhood.pt',map_location='cpu',weights_only=False)['state'].items()}
    def transform(ids):
        native=np.asarray(ccc[ids])[:,mask];activity=native.sum(1)
        profile=native/scale/np.maximum(activity,1e-12)[:,None]
        communication=np.column_stack(((profile-state['profile_center'])/state['profile_scale'],
            .25*(np.log1p(activity)-state['activity_center'])/state['activity_scale']))/state['communication_scale']
        mix=np.asarray(comp[ids]);mix=np.sqrt(np.maximum(mix,0)/np.maximum(mix.sum(1,keepdims=True),1e-12))
        return communication,(mix-state['composition_center'])/state['composition_scale']
    def choose(take,n):
        ids=np.flatnonzero(take);return np.sort(rng.choice(ids,min(n,len(ids)),replace=False))
    def dmat(z):
        z=torch.from_numpy(np.asarray(z,dtype=np.float32));v=(z*z).sum(1)
        return (v[:,None]+v[None,:]-2*z@z.T).clamp_min(0).numpy()
    def spectrum(z):
        ev=np.maximum(np.linalg.eigvalsh(np.cov(np.asarray(z,dtype=np.float64),rowvar=False))[::-1],0)
        p=ev/ev.sum();return dict(PC1=float(p[0]),top5=float(p[:5].sum()),top10=float(p[:10].sum()),
            dimensions_90=int(np.searchsorted(np.cumsum(p),.9)+1),participation_rank=float(1/(p*p).sum()),
            total_variance=float(ev.sum()))
    result=dict(protocol='Saved frozen epoch5 h; no training, relabeling or notebook modifications',
                fit=str(FIT),ROI='Approximate user box; posthoc only',neighbor_candidates='Random subset within each named scope',seed=40700)
    scopes={'all_cells':choose(np.ones(len(a),bool),3000),
            'N1_cancer':choose((labels==1)&(typ=='Cancer cell'),3000),
            'A_core_N1_cancer':choose((labels==1)&(typ=='Cancer cell')&(sample=='crc_a_c'),2500)}
    metrics=[]
    for name,ids in scopes.items():
        print('Audit target/h distances:',name,len(ids),flush=True)
        c,m=transform(ids);z=np.asarray(h[ids]);dh,dc,dm=dmat(z),dmat(c),dmat(m);dj=dc+dm
        ia=rng.integers(len(ids),size=50000);ib=rng.integers(len(ids),size=50000);good=ia!=ib;ia,ib=ia[good],ib[good]
        row=dict(scope=name,cells=len(ids),h_spectrum=spectrum(z),
            CCC_variance=float(((c-c.mean(0))**2).sum(1).mean()),
            composition_variance=float(((m-m.mean(0))**2).sum(1).mean()))
        row['joint_target_variance']=row['CCC_variance']+row['composition_variance']
        for key,d in [('CCC',dc),('composition',dm),('joint',dj)]:
            row[key+'_distance_spearman']=float(spearmanr(dh[ia,ib],d[ia,ib]).statistic)
            row[key+'_pair_distance_quantiles']=np.quantile(d[ia,ib],[.1,.5,.9,.99]).tolist()
            np.fill_diagonal(d,np.inf)
            neighbors=np.argpartition(d,15,axis=1)[:,:15]
            if key=='CCC':
                np.fill_diagonal(dh,np.inf);hn=np.argpartition(dh,15,axis=1)[:,:15]
            row[key+'_neighbor_recall15']=float(np.mean([len(set(x)&set(y))/15 for x,y in zip(hn,neighbors)]))
        row['h_pair_distance_quantiles']=np.quantile(dh[ia,ib],[.1,.5,.9,.99]).tolist()
        row['random_neighbor_recall15']=15/(len(ids)-1)
        result.setdefault('scopes',{})[name]=row
        print(json.dumps(row),flush=True)
    # Exact within-cluster sum of squares and potential gain from splitting the nominated branch.
    sums=np.zeros((12,96));ss=np.zeros(12);counts=np.bincount(labels,minlength=13)[1:]
    for lo in range(0,len(h),32768):
        z=np.asarray(h[lo:lo+32768],dtype=np.float64);ls=labels[lo:lo+len(z)]
        for k in range(1,13):
            v=z[ls==k];sums[k-1]+=v.sum(0);ss[k-1]+=(v*v).sum()
    centers=sums/counts[:,None];sse=ss-(sums*sums).sum(1)/counts
    table=pd.DataFrame({'niche':[f'N{k}' for k in range(1,13)],'cells':counts,'SSE':sse,
                        'SSE_fraction':sse/sse.sum(),'variance_per_cell':sse/counts})
    table.to_csv(OUT/'h_K12_variance.csv',index=False);print(table.to_string(index=False),flush=True)
    take=(labels==1)&roi;nr=take.sum();zr=np.asarray(h[take],dtype=np.float64).mean(0)
    nc=counts[0]-nr;zc=(sums[0]-nr*zr)/nc
    gain=nr*nc/counts[0]*float(((zr-zc)**2).sum())
    result['ROI_binary_split_gain']=dict(N1_roi_cells=int(nr),SSE_reduction=gain,
        fraction_of_all_K12_SSE=float(gain/sse.sum()),fraction_of_N1_SSE=float(gain/sse[0]),
        note='Diagnostic forced split only; no labels changed and no optimization performed')
    contrasts=[]
    for sid in ['crc_a_c','crc_a_m']:
        common=(sample==sid)&(typ=='Cancer cell')&(labels==1)
        groups={'ROI':choose(common&roi,1600),'outside':choose(common&~roi,1600)}
        blocks={}
        for group,ids in groups.items():
            c,m=transform(ids);blocks[group]=dict(h=np.asarray(h[ids]),CCC=c,composition=m,joint=np.column_stack((c,m)))
        for key in ['h','CCC','composition','joint']:
            u=blocks['ROI'][key];v=blocks['outside'][key]
            gap=float(((u.mean(0)-v.mean(0))**2).sum());within=float(((u-u.mean(0))**2).sum(1).mean()+((v-v.mean(0))**2).sum(1).mean())
            contrasts.append(dict(specimen=sid,space=key,centroid_gap_squared=gap,within_pair_variance=within,
                between_vs_within_ratio=1+gap/max(within,1e-20)))
    pd.DataFrame(contrasts).to_csv(OUT/'ROI_same_specimen_geometry.csv',index=False)
    result['ROI_same_specimen_geometry']=contrasts
    # A/B margin comparisons restricted to cancer centers within the SAME N10 class.
    sides={}
    for sid in ['crc_a_m','crc_b_m']:
        ids=choose((labels==10)&(typ=='Cancer cell')&(sample==sid),1800)
        c,m=transform(ids);sides[sid]=dict(h=np.asarray(h[ids]),CCC=c,composition=m,joint=np.column_stack((c,m)))
    result['N10_A_B_margin']={}
    for key in ['h','CCC','composition','joint']:
        u=sides['crc_a_m'][key];v=sides['crc_b_m'][key]
        gap=float(((u.mean(0)-v.mean(0))**2).sum());within=float(((u-u.mean(0))**2).sum(1).mean()+((v-v.mean(0))**2).sum(1).mean())
        result['N10_A_B_margin'][key]=dict(centroid_gap_squared=gap,within_pair_variance=within,between_vs_within_ratio=1+gap/max(within,1e-20))
    (OUT/'audit.json').write_text(json.dumps(result,indent=2))
    print('ROI split gain',result['ROI_binary_split_gain'],flush=True)
    print('ROI contrasts',contrasts,flush=True)
    print('N10 contrasts',result['N10_A_B_margin'],flush=True)
    print('Saved audit:',OUT,flush=True)


if __name__=='__main__':run()
