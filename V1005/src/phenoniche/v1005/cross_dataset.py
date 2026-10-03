import numpy as np
import pandas as pd
from .data import ROOT,json_write
from .evaluation import match_programs

def match_datasets():
    out=ROOT/'outputs/cross_dataset';out.mkdir(parents=True,exist_ok=True)
    a=pd.read_csv(ROOT/'outputs/hbc1/features.csv');b=pd.read_csv(ROOT/'outputs/xenium5k/features.csv')
    keys=['sender','receiver','ligand','receptor']
    joined=a.reset_index().merge(b.reset_index(),on=keys,suffixes=('_a','_b'))
    if len(joined)==0:raise ValueError('No shared retained directed CCC features')
    shared_measurable_count=None
    manifest_a=ROOT/'outputs/hbc1/measurable_lr.csv';manifest_b=ROOT/'outputs/xenium5k/measurable_lr.csv'
    if manifest_a.exists() and manifest_b.exists():
        shared_lr=pd.read_csv(manifest_a).merge(pd.read_csv(manifest_b),on=['ligand','receptor'])[['ligand','receptor']].drop_duplicates()
        shared_measurable_count=len(shared_lr)
        types=sorted((set(a.sender)|set(a.receiver)) & (set(b.sender)|set(b.receiver)))
        domains=pd.MultiIndex.from_product([types,types],names=['sender','receiver']).to_frame(index=False)
        audit=domains.merge(shared_lr,how='cross')
        audit=audit.merge(a[keys].assign(retained_hbc1=True),on=keys,how='left').merge(b[keys].assign(retained_xenium5k=True),on=keys,how='left')
        audit[['retained_hbc1','retained_xenium5k']]=audit[['retained_hbc1','retained_xenium5k']].eq(True)
        audit.to_csv(out/'shared_assay_filter_audit.csv',index=False)
    full_a=np.load(ROOT/'outputs/hbc1/seed_40700_tensor_H.npy')
    full_b=np.load(ROOT/'outputs/xenium5k/seed_40700_tensor_H.npy')
    ha=full_a[:,joined.index_a];hb=full_b[:,joined.index_b]
    coverage=[]
    for ds,full,shared in [('hbc1',full_a,ha),('xenium5k',full_b,hb)]:
        norm=full/np.maximum(np.linalg.norm(full,axis=1,keepdims=True),1e-12)
        within=norm@norm.T
        np.save(out/f'{ds}_within_program_cosine.npy',within)
        for k in range(len(full)):
            coverage.append({'dataset':ds,'program':k+1,'shared_weight_fraction':float(shared[k].sum()/max(full[k].sum(),1e-12)),
              'maximum_other_program_cosine':float(np.max(np.delete(within[k],k))),
              'note':'Nonshared weight reflects panel/filter differences as well as potential biological differences'})
    pd.DataFrame(coverage).to_csv(out/'program_coverage.csv',index=False)
    s,i,j=match_programs(ha,hb)
    result=pd.DataFrame({'hbc1_program':i+1,'xenium5k_program':j+1,'cosine_similarity':s[i,j],
                         'n_shared_retained_CCC':len(joined)})
    result['evidence_qualification']='Descriptive only: inspect shared feature count and weight coverage; not biological validation'
    # Descriptive fixed flag only; not a significance claim or a tuning criterion.
    result['descriptive_status']=np.where(result.cosine_similarity>=.8,'high_similarity_on_shared_features','lower_similarity_on_shared_features')
    if len(joined)<=2:result['descriptive_status']='insufficient_shared_dimensions_for_program_claim'
    result.to_csv(out/'matched_programs.csv',index=False);np.save(out/'similarity.npy',s)
    joined.to_csv(out/'shared_feature_mapping.csv',index=False)
    json_write(out/'audit.json',{'n_shared_features':len(joined),'n_shared_measurable_LR':shared_measurable_count,'mean_matched_cosine':float(s[i,j].mean()),
          'limitation':'Matching only shared retained measurable features; absent panel features do not demonstrate dataset-specific biology; provisional 5K labels.'})
    return result
