"""Ground truth is accessed only here, after training and unsupervised clustering."""
from pathlib import Path
import json
import numpy as np,pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score,f1_score,jaccard_score,accuracy_score,roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from ..gnn.losses import ccc_loss,composition_loss
import torch

def label_mapping(predicted,truth,rows):
    p=np.unique(predicted);t=np.unique(truth[rows]);table=np.asarray([[(predicted[rows]==a)[truth[rows]==b].sum() for b in t] for a in p])
    i,j=linear_sum_assignment(-table);mapping={int(p[a]):int(t[b]) for a,b in zip(i,j)}
    for a,v in enumerate(p):
        if int(v) not in mapping:mapping[int(v)]=int(t[table[a].argmax()])
    return mapping

def mapped(predicted,mapping):return np.asarray([mapping[int(v)] for v in predicted])


def partition_scores(truth,pred,aligned,rows,ab,edge_train):
    edge=np.isin(truth,[2,3]);erows=rows[edge[rows]]
    overall_ok=len(np.unique(truth[rows]))>=2
    edge_ok=len(np.unique(edge[rows]))==2
    ab_ok=len(np.unique(truth[erows]))==2 and len(edge_train)>0
    return {'overall_ARI':adjusted_rand_score(truth[rows],pred[rows]) if overall_ok else np.nan,
            'overall_NMI':normalized_mutual_info_score(truth[rows],pred[rows]) if overall_ok else np.nan,
            'edge_F1':f1_score(edge[rows],np.isin(aligned[rows],[2,3]),zero_division=0) if edge_ok else np.nan,
            'edge_IoU':jaccard_score(edge[rows],np.isin(aligned[rows],[2,3]),zero_division=0) if edge_ok else np.nan,
            'edge_AB_ARI':adjusted_rand_score(truth[erows],pred[erows]) if ab_ok else np.nan,
            'edge_AB_accuracy':accuracy_score(truth[erows],mapped(pred[erows],ab)) if ab_ok else np.nan,
            'overall_evaluable':overall_ok,'edge_detection_evaluable':edge_ok,'edge_AB_evaluable':ab_ok,'n':len(rows),'n_true_edge':len(erows)}

def composition_audit(adata,data,output):
    truth=adata.obs.true_niche.to_numpy();edge=np.flatnonzero(np.isin(truth,[2,3]));x=np.asarray(data.composition_target[edge]);y=(truth[edge]==3).astype(int)
    groups=adata.obs.mirror_pair.to_numpy()[edge];n_splits=min(5,len(np.unique(groups)))
    prediction=np.zeros(len(edge));auc=np.nan
    if n_splits>=2 and len(np.unique(y))==2:
        for tr,te in GroupKFold(n_splits=n_splits).split(x,y,groups):
            model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,random_state=0)).fit(x[tr],y[tr]);prediction[te]=model.predict_proba(x[te])[:,1]
        auc=float(roc_auc_score(y,prediction))
    records=[];radius=adata.obs.radius_um.to_numpy();lo=radius[edge].min();hi=radius[edge].max()+1e-6
    bins=np.linspace(lo,hi,9)
    for band in range(8):
        a=(truth==2)&(radius>=bins[band])&(radius<bins[band+1]);b=(truth==3)&(radius>=bins[band])&(radius<bins[band+1])
        if not a.any() or not b.any():continue
        ma=np.asarray(data.composition_target[a]).mean(0);mb=np.asarray(data.composition_target[b]).mean(0)
        for t,name in enumerate(data.categories):records.append({'radial_bin':band,'cell_type':name,'A':float(ma[t]),'B':float(mb[t]),'difference':float(ma[t]-mb[t])})
    frame=pd.DataFrame(records);frame.to_csv(Path(output)/'radial_composition_audit.csv',index=False)
    pairs=adata.obs.mirror_pair.to_numpy();n=len(adata)//2
    if not np.array_equal(pairs[:n],pairs[n:]):raise ValueError('Mirror identity lost')
    pair_difference=float(np.max(np.abs(np.asarray(data.composition_target[:n])-np.asarray(data.composition_target[n:]))))
    strict=[]
    if adata.uns['injection_mode']=='spatial_pairing_only':
        xfull=np.asarray(adata.X);bands=np.floor((radius-radius[edge].min())/(radius[edge].max()-radius[edge].min()+1e-6)*8).astype(int)
        # Gene-wise marginals and complete row-profile multisets are compared
        # in the exact injection strata, not pooled across cell types.
        # Actual permutation strata are recorded by generator geometry in config.
        cfg=json.loads((Path(output)/'config.json').read_text());bands=np.clip(((radius-cfg['core_radius_um'])/cfg['edge_width_um']*cfg['radial_bins']).astype(int),0,cfg['radial_bins']-1)
        for sample in adata.obs.sample_id.unique():
            for band in np.unique(bands[edge]):
                for ct in data.categories:
                    keep=(adata.obs.sample_id.to_numpy()==sample)&(bands==band)&(adata.obs.cell_type.to_numpy()==ct)
                    a=np.flatnonzero(keep&(truth==2));b=np.flatnonzero(keep&(truth==3))
                    if len(a):strict.append(float(np.max(np.abs(np.sort(xfull[a],axis=0)-np.sort(xfull[b],axis=0)))))
    result={'composition_only_grouped_CV_AUC':auc,'CV_groups':'mirror pairs stay together; diagnostic classifier uses true A/B labels only in evaluation','max_mirror_local_composition_difference':pair_difference,'radial_composition_mean_abs_difference':float(frame.difference.abs().mean()) if len(frame) else None,'strict_mode_max_gene_marginal_difference':max(strict) if strict else None}
    pd.DataFrame({'cell_id':adata.obs.cell_id.iloc[edge].to_numpy(),'true_B':y,'cross_validated_B_probability':prediction}).to_csv(Path(output)/'composition_AUC_predictions.csv',index=False)
    return result

def evaluate(adata,data,labels,arrays,h,b,ae,output):
    output=Path(output);truth=adata.obs.true_niche.to_numpy();train,test=data.rows[0],data.rows[2];edge=np.isin(truth,[2,3]);records=[];maps={}
    for method,pred in labels.items():
        mapping=label_mapping(pred,truth,train);aligned=mapped(pred,mapping);maps[method]=mapping
        edge_train=train[edge[train]];edge_test=test[edge[test]]
        ab=label_mapping(pred,truth,edge_train) if len(edge_train) else mapping
        for split,rows in [('all_transductive',np.arange(len(truth))),('heldout_test',test)]:
            records.append({'method':method,'evaluation':split,'K':len(np.unique(pred)),**partition_scores(truth,pred,aligned,rows,ab,edge_train)})
    pd.DataFrame(records).to_csv(output/'metrics.csv',index=False)
    (output/'evaluation_label_mapping.json').write_text(json.dumps({'mapping_fit_rows':'training rows only; evaluation mapping never used by clustering','mappings':maps},indent=2))
    true_h=np.load(output/'ground_truth/MI_dictionary.npy');true_z=np.load(output/'ground_truth/MI_activity.npy')
    cosine=(true_h@h.T)/np.maximum(np.linalg.norm(true_h,axis=1)[:,None]*np.linalg.norm(h,axis=1)[None,:],1e-12)
    initial_path=output/'gnn/initial_H.npy';initial=np.load(initial_path) if initial_path.exists() else None
    initial_cosine=(true_h@initial.T)/np.maximum(np.linalg.norm(true_h,axis=1)[:,None]*np.linalg.norm(initial,axis=1)[None,:],1e-12) if initial is not None else None
    mi,program=linear_sum_assignment(-cosine);matches=[]
    for i,p in zip(mi,program):
        support=set(np.flatnonzero(true_h[i]));top=set(np.argsort(-h[p],kind='stable')[:len(support)])
        for name,rows in [('test_all',test),('test_edge',test[edge[test]])]:
            rho=float(spearmanr(true_z[rows,i],np.asarray(arrays['Z'][rows,p])).statistic) if len(rows)>2 and np.ptp(true_z[rows,i])>0 and np.ptp(np.asarray(arrays['Z'][rows,p]))>0 else np.nan
            matches.append({'MI':i+1,'program':p+1,'scope':name,'dictionary_cosine':float(cosine[i,p]),'initial_dictionary_cosine':float(initial_cosine[i,p]) if initial_cosine is not None else np.nan,'dictionary_cosine_gain':float(cosine[i,p]-initial_cosine[i,p]) if initial_cosine is not None else np.nan,'top_LR_precision':len(support&top)/max(len(support),1),'activity_Spearman':rho,'match_policy':'one-to-one dictionary matching only; no test activity used to choose programs'})
    pd.DataFrame(matches).to_csv(output/'MI_matches.csv',index=False)
    use_composition=json.loads((output/'config.json').read_text())['lambda_comp']>0
    metrics={};weights=0.;ccc=0.;comp=0.;ae_ccc=0.
    for start in range(0,len(test),512):
        rows=test[start:start+512];target=torch.from_numpy(data.targets[rows]);reconstruction=torch.from_numpy(np.asarray(arrays['Z'][rows])@h)
        ccc+=float(ccc_loss(reconstruction,target,data.feature_scale,data.feature_mask))*len(rows)
        if use_composition:comp+=float(composition_loss(torch.from_numpy(np.asarray(arrays['P'][rows])@b),torch.from_numpy(np.asarray(data.composition_target[rows]))))*len(rows)
        ae_ccc+=float(ccc_loss(torch.from_numpy(np.asarray(ae['Z'][rows])@ae['H']),target,data.feature_scale,data.feature_mask))*len(rows);weights+=len(rows)
    metrics.update(gnn_test_CCC_Huber=ccc/weights,ae_test_CCC_Huber=ae_ccc/weights)
    if use_composition:metrics['gnn_test_composition_KL']=comp/weights
    split_counts=pd.crosstab(pd.Series(data.folds,name='fold'),pd.Series(truth,name='true_niche'))
    split_counts.to_csv(output/'evaluation_split_coverage.csv')
    metrics['test_contains_edge']=bool(edge[test].any())
    metrics['best_epoch']=json.loads((output/'gnn/heldout_metrics.json').read_text())['best_epoch']
    metrics.update(composition_audit(adata,data,output));metrics['interpretation']='Synthetic recovery diagnostics; no biological causality or clinical phenotype claim. Raw cluster IDs are saved independently from evaluation mappings.'
    (output/'diagnostics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    return pd.DataFrame(records),metrics
