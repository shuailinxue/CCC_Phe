"""Read-only geometry/statistics audit; never call prepare(), fit(), or change caches."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[1]
OUT=Path('/home/nas3/biod/xueshuailin/CCC_Phe/workspace/outputs/tnbc_prepared_inputs_audit')
BASE=Path('/home/nas3/biod/xueshuailin/CCC_Phe/workspace/outputs/tnbc_communication_phenotype')


def main():
    OUT.mkdir(exist_ok=True,parents=True);root=BASE/'input';summary=pd.read_csv(root/'sample_summary.csv')
    meta=json.loads((root/'catalog.json').read_text());prep=np.load(root/'preprocessing.npz');mask=prep['feature_mask'];features=pd.read_csv(root/'features.csv')
    sums=np.zeros(len(mask));squares=sums.copy();nz=sums.copy();available=sums.copy();records=[];signatures=set();rng=np.random.default_rng(40700)
    geometry=[];contributions=[]
    for idx,row in enumerate(summary.itertuples()):
        d=root/row.sample;xy=np.load(d/'coordinates_um.npy',mmap_mode='r');m=np.load(d/'members.npy',mmap_mode='r');anchor=np.load(d/'anchor_weights.npy',mmap_mode='r');types=np.load(d/'types.npy',mmap_mode='r')
        ids=np.sort(rng.choice(len(xy),min(1024,len(xy)),replace=False));local=np.asarray(m[ids]);coords=np.asarray(xy[local]);center=np.asarray(xy[ids])
        r=np.linalg.norm(coords-center[:,None],axis=2);nn=np.where(local==ids[:,None],np.inf,r).min(1);radius=r.max(1)
        self_counts=(local==ids[:,None]).sum(1);dups=np.array([30-len(set(v)) for v in local]);distance2=r*r
        expected=np.exp(-distance2/(2*20.**2));err=float(np.max(np.abs(expected-anchor[ids])))
        w=np.exp(-np.sum((coords[:,:,None,:]-coords[:,None,:,:])**2,axis=3)/(2*20.**2))*np.asarray(anchor[ids])[:,:,None]*np.asarray(anchor[ids])[:,None,:]
        w[:,np.arange(30),np.arange(30)]=0
        degree=w.sum(2);eff=degree.sum(1)**2/np.maximum((degree*degree).sum(1),1e-30)
        lengths=np.linalg.norm(coords[:,:,None,:]-coords[:,None,:,:],axis=3)
        long_weight=(w*(lengths>60)).sum()/max(w.sum(),1e-30)
        nearest_check=None
        if idx in (0,23,46,69,92):
            qdist,_=cKDTree(xy).query(center[:32],k=30)
            nearest_check=float(np.max(np.abs(np.sort(r[:32],axis=1)-qdist)))
        s=np.load(d/'ccc_statistics.npz');elig=np.load(d/'eligible_features.npy').astype(bool);n=len(xy)
        checks={k:bool(np.isfinite(s[k]).all() and (s[k]>=0).all()) for k in s.files}
        assert all(checks.values()) and (s['nonzero']<=n).all()
        assert np.allclose(s['available'],elig*n)
        total= s['sum']*elig;ss=s['square']*elig;nnz=s['nonzero']*elig
        sums+=total;squares+=ss;nz+=nnz;available+=s['available']
        contributions.append(total)
        sig=json.loads((d/'statistics_complete.json').read_text())['signature'];signatures.add(sig)
        mono=(types[local]==types[local][:,:1]).all(1)
        records.append({'sample':row.sample,'cells':n,'checked_centers':len(ids),'bounds_ok':bool(((local>=0)&(local<n)).all()),
            'self_once':bool((self_counts==1).all()),'duplicate_neighbors':int(dups.sum()),'nonfinite_coords_in_sample':int((~np.isfinite(coords)).sum()),
            'duplicate_position_fraction':float((nn<1e-4).mean()),'nearest_um_p50':float(np.median(nn)),
            'radius_um_p10':float(np.quantile(radius,.1)),'radius_um_p50':float(np.median(radius)),
            'radius_um_p90':float(np.quantile(radius,.9)),'radius_um_p99':float(np.quantile(radius,.99)),
            'radius_gt60_fraction':float((radius>60).mean()),'physical_weight_over60um_fraction':float(long_weight),
            'effective_cells_p50':float(np.median(eff)),'anchor_formula_maxerror':err,
            'KNN_distance_check_maxerror':nearest_check,'single_lineage_neighborhood_fraction':float(mono.mean()),
            'eligible_LR':int(elig.sum()/25),'effective_features_in_sample':int((elig&mask).sum()),
            'native_CCC_sum_per_center':float(total.sum()/n),'positive_CCC_fraction_eligible':float(nnz[elig].sum()/max(n*elig.sum(),1)),
            'input_stats_finite_nonnegative':True})
        geometry.append(pd.DataFrame({'sample':row.sample,'source_row':ids,'radius_um':radius,'nearest_um':nn,'effective_cells':eff}))
        if (idx+1)%20==0:print('AUDITED',idx+1,flush=True)
    records=pd.DataFrame(records);records.to_csv(OUT/'sample_geometry_and_CCC.csv',index=False)
    geom=pd.concat(geometry,ignore_index=True);geom.to_csv(OUT/'sampled_geometry.csv.gz',index=False)
    mean=sums/np.maximum(available,1);ms=squares/np.maximum(available,1);rho=nz/np.maximum(available,1)
    assert np.allclose(mean,prep['feature_mean'],rtol=1e-5,atol=1e-8)
    assert np.allclose(np.maximum(np.sqrt(ms),1e-6),prep['feature_scale'],rtol=1e-5,atol=1e-8)
    assert np.allclose(rho,prep['feature_nonzero_fraction'],rtol=1e-5,atol=1e-8)
    contribution=np.stack(contributions)
    dominance=contribution.max(0)/np.maximum(contribution.sum(0),1e-30)
    features['effective']=mask;features['mean']=mean;features['RMS']=np.sqrt(ms);features['positive_fraction']=rho
    features['available_patient_count']=np.stack([np.load(root/s/'eligible_features.npy') for s in summary['sample']]).sum(0)
    features['largest_patient_signal_share']=dominance
    features.to_csv(OUT/'CCC_feature_audit.csv',index=False)
    patient_fraction=contribution.sum(1)/contribution.sum()
    records['share_of_native_CCC_sum']=patient_fraction;records.to_csv(OUT/'sample_geometry_and_CCC.csv',index=False)
    fitted=pd.read_csv(BASE/'dual/patient_features/patient_features.csv')
    correlations=[]
    for col in ['radius_um_p50','effective_cells_p50','single_lineage_neighborhood_fraction','eligible_LR','native_CCC_sum_per_center','positive_CCC_fraction_eligible']:
        t=fitted.merge(records[['sample',col]],on='sample');correlations.append({'metric':col,'rho_with_N3_fraction':float(spearmanr(t[col],t.cell_fraction__N3).statistic)})
    pd.DataFrame(correlations).to_csv(OUT/'geometry_CCC_vs_N3.csv',index=False)
    def quantile(v):return dict(zip(['min','p10','p50','p90','max'],map(float,np.quantile(v,[0,.1,.5,.9,1]))))
    report={'samples':len(records),'cells':int(records.cells.sum()),'sampled_centers':len(geom),
        'all_sampled_members_in_bounds':bool(records.bounds_ok.all()),'self_once':bool(records.self_once.all()),
        'duplicate_neighbor_entries':int(records.duplicate_neighbors.sum()),'CCC_signature_count':len(signatures),
        'global_CCC_moments_reaggregated_match':True,'radius_um':quantile(geom.radius_um),'nearest_um':quantile(geom.nearest_um),
        'effective_cells':quantile(geom.effective_cells),'sample_median_radius_range_um':[float(records.radius_um_p50.min()),float(records.radius_um_p50.max())],
        'eligible_LR':quantile(records.eligible_LR),'total_LR':len(meta['links']),'effective_CCC_features':int(mask.sum()),
        'sample_native_CCC_mean':quantile(records.native_CCC_sum_per_center),
        'sample_positive_CCC_fraction':quantile(records.positive_CCC_fraction_eligible),
        'effective_feature_RMS':quantile(features.loc[mask,'RMS']),
        'effective_feature_available_patients':quantile(features.loc[mask,'available_patient_count']),
        'effective_feature_largest_patient_share':quantile(dominance[mask]),
        'global_native_CCC_sum_top5_patient_share':float(np.sort(patient_fraction)[-5:].sum()),
        'cell_count_top5_patient_share':float(np.sort(records.cells.to_numpy()/records.cells.sum())[-5:].sum()),
        'geometry_N3_correlations':correlations,'read_only':True,'sampling':'up to1024 centers/sample; exact fresh-KNN checks32 centers in5 samples; CCC moments reaggregated across all93 samples'}
    (OUT/'summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2));print('TOP_CCC_SAMPLES');print(records.nlargest(5,'share_of_native_CCC_sum')[['sample','cells','eligible_LR','native_CCC_sum_per_center','share_of_native_CCC_sum']].to_string(index=False))


if __name__=='__main__':main()
