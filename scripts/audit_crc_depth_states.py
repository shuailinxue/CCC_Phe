"""Read-only depth/area and LR sensitivity audit of saved pure-cancer states.

Panel-total scaling below is a diagnostic, not a replacement for native CCC.
No model, target cache, labels, or notebook outputs are changed.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(key,'6')
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import anndata as ad
import h5py
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

PROJECT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECT/'src'))
BASE=Path('/home/nas3/biod/xueshuailin/CCC_Phe/workspace/outputs/crc_neutrophil_xenium/joint_five')
ROOT=Path('/home/nas3/biod/xueshuailin/CCC_Phe/workspace/data/crc_neutrophil_xenium')
OUT=BASE/'tumor_niche_analysis/depth_audit'


def run():
    from phenoniche.ccc.lr_atlas import load_lr_atlas,LRAtlas
    from phenoniche.ccc.pairwise import _side_expression
    OUT.mkdir(exist_ok=True);rng=np.random.default_rng(40700)
    a=ad.read_h5ad(ROOT/'processed/CRC_selected_five.h5ad')
    receipt=json.loads((BASE/'figures/umap_leiden_control.json').read_text());fit=Path(receipt['KMeans']).parents[1]
    matrix=Path(json.loads((fit/'complete.json').read_text())['matrices'])
    labels=np.load(Path(receipt['Leiden'])/'labels.npy')+1;sample=a.obs['sample'].to_numpy();types=a.obs.cell_type.to_numpy()
    total=np.asarray(a.X.sum(1)).ravel();detected=np.asarray((a.X>0).sum(1)).ravel()
    # Whitelist technical metadata only. No clinical, known niche or outcome fields.
    metadata={};source_rows=a.obs.source_row.to_numpy()
    with h5py.File(ROOT/'crca_xenium.h5ad','r') as f:
        for key in ['cell_area','nucleus_area','transcript_density','control_probe_counts','control_codeword_counts','transcript_counts']:
            v=f['obs'][key]
            if isinstance(v,h5py.Dataset):metadata[key]=np.asarray(v)[source_rows]
        for key in ['slide','segmentation_method']:
            g=f['obs'][key]
            def values(dataset):
                return np.asarray(dataset.asstr()) if h5py.check_string_dtype(dataset.dtype) else np.asarray(dataset).astype(str)
            if isinstance(g,h5py.Group):metadata[key]=values(g['categories'])[np.asarray(g['codes'])[source_rows]]
            else:metadata[key]=values(g)[source_rows]
    if not np.array_equal(total,metadata['transcript_counts']):raise ValueError('Author source row/count mismatch')
    frame=pd.DataFrame(dict(label=labels,specimen=sample,cell_type=types,total=total,detected=detected,**metadata))
    frame['counts_per_area']=total/np.maximum(frame.cell_area,1e-6)
    rows=[]
    for k in [4,8,10]:
        for sid in ['ALL']+list(pd.unique(sample)):
            take=(labels==k)&(types=='Cancer cell')
            if sid!='ALL':take&=sample==sid
            if not take.any():continue
            z=frame.loc[take];r=dict(niche=f'L{k}',specimen=sid,cells=len(z))
            for key in ['total','detected','cell_area','nucleus_area','counts_per_area','control_probe_counts','control_codeword_counts','transcript_density']:
                r[key+'_median']=float(z[key].median());r[key+'_q25']=float(z[key].quantile(.25));r[key+'_q75']=float(z[key].quantile(.75))
                r[key+'_missing']=int(z[key].isna().sum())
            r['slides']=';'.join(f'{s}:{n}' for s,n in z.slide.value_counts().items())
            r['segmentation']=';'.join(f'{s}:{n}' for s,n in z.segmentation_method.value_counts().items())
            rows.append(r)
    summary=pd.DataFrame(rows);summary.to_csv(OUT/'depth_area_summary.csv',index=False)
    print(summary[['niche','specimen','cells','total_median','detected_median','cell_area_median','counts_per_area_median','slides']].to_string(index=False),flush=True)
    # Variance association is descriptive on the same training specimens, not a causal batch estimate.
    take=np.flatnonzero(np.isin(labels,[4,8,10])&(types=='Cancer cell'))
    eligible=np.isfinite(frame.loc[take,['total','detected','cell_area','nucleus_area']].to_numpy()).all(1)
    excluded=int((~eligible).sum());take=take[eligible]
    ids=np.sort(rng.choice(take,min(18000,len(take)),replace=False));h=np.asarray(np.load(Path(receipt['KMeans'])/'joint_embedding.npy',mmap_mode='r')[ids])
    train,test=train_test_split(np.arange(len(ids)),test_size=.3,random_state=40700)
    results={}
    for name,cols in [('panel_count_only',['total']),('count_and_genes',['total','detected']),('count_gene_area',['total','detected','cell_area','nucleus_area'])]:
        x=np.log1p(frame.loc[ids,cols].to_numpy());x=StandardScaler().fit(x[train]).transform(x)
        model=Ridge(alpha=1.).fit(x[train],h[train]);pred=model.predict(x[test])
        results[name]=dict(test_variance_weighted_h_R2=float(r2_score(h[test],pred,multioutput='variance_weighted')),
            complete_case_cells=len(take),excluded_missing_area_cells=excluded,
            note='Cell-row split, exploratory technical association; biological variation can drive these covariates too')
    # Compare unlogged depth-scaled LR values using identical physical members and weights.
    features=pd.read_csv(matrix/'features.csv');active=np.load(fit/'feature_mask.npy')
    selected=features.sender.eq('Cancer cell')&features.receiver.eq('Cancer cell')&active
    columns=np.flatnonzero(selected);ff=features.loc[selected].reset_index(names='feature_index')
    atlas=load_lr_atlas(PROJECT/'data/commuspace_human_lr_atlas.tsv');catalog={v.lr_id:v for v in atlas.interactions}
    links=tuple(catalog[q] for q in ff.lr_id);atlas=LRAtlas(links,{},atlas.source_path)
    members=np.load(matrix/'members.npy',mmap_mode='r');aw=np.load(matrix/'anchor_weights.npy',mmap_mode='r');xy=np.asarray(a.obsm['spatial'])
    groups={}
    for k in [4,8,10]:
        available=np.flatnonzero((labels==k)&(types=='Cancer cell'));groups[f'L{k}']=np.sort(rng.choice(available,min(1800,len(available)),replace=False))
    for k in [8,10]:
        available=np.flatnonzero((labels==k)&(types=='Cancer cell')&(sample=='crc_a_c'))
        groups[f'L{k}_Acore']=np.sort(rng.choice(available,min(1400,len(available)),replace=False))
    arrays={};rows=[];gene_rows=[]
    for name,centers in groups.items():
        print('Same-member native/panel-depth CCC sensitivity:',name,len(centers),'centers',len(links),'LR axes',flush=True)
        all_members=np.asarray(members[centers]);unique,inverse=np.unique(all_members,return_inverse=True);local=inverse.reshape(all_members.shape)
        expr=a.X[unique].toarray().astype(np.float32);l,r=_side_expression(expr,tuple(a.var_names),atlas)
        factor=10000/np.maximum(total[unique],1)
        outputs={}
        for mode in ['native','panel_total']:
            ls=np.sqrt(l);rs=np.sqrt(r)
            if mode=='panel_total':ls*=np.sqrt(factor[:,None]);rs*=np.sqrt(factor[:,None])
            values=[]
            for start in range(0,len(centers),128):
                stop=min(start+128,len(centers));mem=all_members[start:stop];idx=local[start:stop];coords=xy[mem]
                w=np.asarray(aw[centers[start:stop]]);w=w[:,:,None]*w[:,None,:]*np.exp(-((coords[:,:,None]-coords[:,None,:])**2).sum(-1)/(2*20.**2))
                valid=(types[mem]=='Cancer cell');w*=valid[:,:,None]*valid[:,None,:]*(mem[:,:,None]!=mem[:,None,:])
                op=w.sum((1,2));signal=np.einsum('biq,bij,bjq->bq',ls[idx],w,rs[idx],optimize=True)/(op[:,None]+1e-4)
                values.append(signal.astype(np.float32))
            outputs[mode]=np.concatenate(values)
        native_saved=np.asarray(np.load(matrix/'ccc.npy',mmap_mode='r')[centers])[:,columns]
        error=float(np.max(np.abs(outputs['native']-native_saved)))
        if not np.allclose(outputs['native'],native_saved,rtol=3e-5,atol=3e-5):raise ValueError('Diagnostic native CCC did not reproduce saved target')
        arrays[name+'_native']=outputs['native'];arrays[name+'_panel_total']=outputs['panel_total'];arrays[name+'_centers']=centers
        for j,f in ff.iterrows():
            rows.append(dict(group=name,ligand=f.ligand,receptor=f.receptor,feature_index=int(f.feature_index),
                cells=len(centers),native_mean=float(outputs['native'][:,j].mean()),normalized_mean=float(outputs['panel_total'][:,j].mean()),
                positive_fraction=float((outputs['native'][:,j]>0).mean()),native_check_max_error=error))
        normalized=expr*(10000/np.maximum(total[unique,None],1))
        for j,gene in enumerate(a.var_names):gene_rows.append(dict(group=name,gene=gene,native_mean=float(expr[:,j].mean()),panel_normalized_mean=float(normalized[:,j].mean())))
    table=pd.DataFrame(rows);table.to_csv(OUT/'CCC_depth_sensitivity.csv',index=False)
    pd.DataFrame(gene_rows).to_csv(OUT/'member_expression_depth_sensitivity.csv',index=False)
    scale=np.load(fit/'feature_scale.npy')[columns];contrasts=[]
    for left,right in [('L4','L8'),('L4','L10'),('L8','L10'),('L8_Acore','L10_Acore')]:
        for mode in ['native','panel_total']:
            u=arrays[left+'_'+mode];v=arrays[right+'_'+mode]
            # Relative profiles use a single fixed feature weighting in both modes.
            u=u/np.maximum(u.sum(1,keepdims=True),1e-12)/scale
            v=v/np.maximum(v.sum(1,keepdims=True),1e-12)/scale
            gap=float(((u.mean(0)-v.mean(0))**2).sum());within=float(((u-u.mean(0))**2).sum(1).mean()+((v-v.mean(0))**2).sum(1).mean())
            contrasts.append(dict(left=left,right=right,mode=mode,profile_between_within_ratio=1+gap/max(within,1e-20)))
    pd.DataFrame(contrasts).to_csv(OUT/'CCC_relative_profile_contrasts.csv',index=False)
    report=dict(protocol='Read-only audit; no training/cache/labels/notebook changes',groups=list(groups),
        diagnostic_normalization='Unlogged CP10k per measured panel total, same LR subunits, members, sigma20um, pair weights and opportunity denominator; sensitivity only',
        encoder_node_input='log1p CP10k, global per-gene mean/std, appended log1p(total)/10',
        encoder_edge_input='Native-count sqrt(L*R) followed by log1p and learned edge projection',
        CCC_target='Native-count weighted sqrt(L*R)/(opportunity+1e-4); feature RMS is shared across samples, not per-cell depth correction',
        model_depth_association=results,CCC_profile_contrasts=contrasts,
        limits='Counts/area can vary biologically; no causal separation of batch and biology. Panel-total scaling is not asserted to be the correct normalization. Cell-level splits are not independent patient validation.')
    (OUT/'audit.json').write_text(json.dumps(report,indent=2));np.savez_compressed(OUT/'diagnostic_CCC.npz',**arrays)
    print('Depth association',results,flush=True);print('CCC profile contrasts',contrasts,flush=True)
    print('Saved:',OUT,flush=True)


if __name__=='__main__':run()
