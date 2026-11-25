"""Outcome-blind breast-atlas annotation; broad lineage before epithelial state.

Comparable multinomial probabilities replace the old hard intersection of two
labels. Low confidence remains visible. Reference donor holdout evaluates the
classifier, not the accuracy of inferred single-cell RNA in a new ST section.
"""
from pathlib import Path
import json,hashlib,pickle
import numpy as np,pandas as pd,scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import balanced_accuracy_score,f1_score
from .genes import MARKERS
from .model import load_upstream


def log_normalize(x):
    if sp.issparse(x):
        x=x.tocsr().astype(np.float32);x=sp.diags(10000/np.maximum(np.asarray(x.sum(1)).ravel(),1e-8))@x;x.data=np.log1p(x.data);return x.toarray()
    x=np.maximum(np.asarray(x,dtype=np.float32),0);return np.log1p(x/np.maximum(x.sum(1,keepdims=True),1e-8)*10000)


def decisions(probabilities,classes,marker_scores,marker_floors,minimum=.5,margin=.1):
    order=np.argsort(probabilities,axis=1);best=order[:,-1];confidence=probabilities[np.arange(len(best)),best]
    difference=confidence-probabilities[np.arange(len(best)),order[:,-2]]
    support=marker_scores[np.arange(len(best)),best]>=marker_floors[best]*.5
    accepted=(confidence>=minimum)&(difference>=margin)&support
    labels=np.asarray(classes,dtype=object)[best];labels[~accepted]='Unknown'
    return labels,confidence,difference,support


def prepare_reference(c,genes,out):
    import anndata as ad
    load_upstream(c['phenomap_repo'])
    from phenomap.cell_annotation.markers import map_reference_cell_type
    out=Path(out);path=out/'annotation_reference.pkl'
    signature={'genes':hashlib.sha256('\n'.join(genes).encode()).hexdigest(),'reference':str(c['reference_atlas']),
        'reference_bytes':Path(c['reference_atlas']).stat().st_size,'reference_mtime':Path(c['reference_atlas']).stat().st_mtime_ns,'code':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'seed':c['seed'],'cells_per_class':c['reference_cells_per_class']}
    if path.exists():
        with path.open('rb') as h:bundle=pickle.load(h)
        if bundle['signature']==signature:return bundle
        raise ValueError('Annotation reference signature changed; choose a separate output_root')
    a=ad.read_h5ad(c['reference_atlas'],backed='r');mapping=a.obs.celltype_major.map(map_reference_cell_type)
    broad=mapping.replace({'Cancer_epithelial':'Epithelial','Normal_epithelial':'Epithelial'})
    rng=np.random.default_rng(c['seed']);rows=[]
    for label in sorted(broad.dropna().unique()):
        ids=np.flatnonzero(broad.eq(label));rows.extend(rng.choice(ids,min(len(ids),c['reference_cells_per_class']),replace=False))
    rows=np.sort(rows);symbols=a.var.feature_name.astype(str).str.upper() if 'feature_name' in a.var else pd.Series(a.var_names,index=a.var_names)
    lookup={g:i for i,g in enumerate(symbols) if g in genes};common=[g for g in genes if g in lookup]
    if len(common)<100:raise ValueError('Insufficient annotation reference gene overlap')
    # Backed AnnData has well-defined original gene order; dense slice <100MB.
    subset=a[rows].to_memory();raw=subset.X[:,[lookup[g] for g in common]].tocsr().astype(np.float32)
    if c['reference_is_log1p']:
        if raw.data.max()>25:raise ValueError('Reference expression is not expected log1p data')
        raw.data=np.expm1(raw.data)
    x=log_normalize(raw);labels=broad.iloc[rows].to_numpy();fine=mapping.iloc[rows].to_numpy()
    donors=a.obs.donor_id.astype(str).iloc[rows].to_numpy();a.file.close()
    train,val=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=c['seed']).split(x,labels,donors))
    mean=x[train].mean(0);sd=np.maximum(x[train].std(0),.3)
    z=np.clip((x-mean)/sd,-10,10)
    classifier=LogisticRegression(C=.2,class_weight='balanced',max_iter=500).fit(z[train],labels[train])
    prediction=classifier.predict(z[val]);validation={'split':'Reference donor holdout; never clinical endpoint or ST niche selection',
        'train_donors':len(set(donors[train])),'validation_donors':len(set(donors[val])),
        'balanced_accuracy':float(balanced_accuracy_score(labels[val],prediction)),
        'macro_F1':float(f1_score(labels[val],prediction,average='macro')),'reference_overlap_genes':len(common)}
    # Refit after evaluating the held-out donor protocol; deployed model uses all sampled donors.
    classifier.fit(z,labels)
    epi=np.isin(fine,['Cancer_epithelial','Normal_epithelial'])
    epithelial=LogisticRegression(C=.1,class_weight='balanced',max_iter=500).fit(z[epi],fine[epi]) if len(set(fine[epi]))==2 else None
    marker_indices=[];floors=[]
    for typ in classifier.classes_:
        ids=[common.index(g) for g in MARKERS[typ] if g in common]
        if len(ids)<2:raise ValueError(f'Insufficient independent lineage markers for {typ}')
        marker_indices.append(ids);scores=x[:,ids].mean(1);floors.append(float(np.quantile(scores[labels==typ],.05)))
    bundle={'signature':signature,'genes':common,'mean':mean,'sd':sd,'classifier':classifier,'epithelial':epithelial,
        'marker_indices':marker_indices,'marker_floors':np.asarray(floors),'validation':validation}
    with path.open('wb') as h:pickle.dump(bundle,h)
    (out/'annotation_reference_audit.json').write_text(json.dumps({**signature,**validation,
        'classes':list(classifier.classes_),'markers':{t:[common[j] for j in ids] for t,ids in zip(classifier.classes_,marker_indices)},
        'decision':'Multinomial confidence≥0.5, margin≥0.1, reference-calibrated marker support; no hard two-method intersection; epithelial subtype only at confidence≥0.7'},indent=2))
    return bundle


def annotate(expression,genes,bundle,index,c,measured_genes=None):
    columns=pd.Index(genes).get_indexer(bundle['genes'])
    if (columns<0).any():raise ValueError('Annotation gene order mismatch')
    x=log_normalize(expression[:,columns]);z=np.clip((x-bundle['mean'])/bundle['sd'],-10,10)
    if measured_genes is not None:
        measured=np.asarray(measured_genes,bool)[columns]
        # Missing source columns are unobserved, not biological zero evidence.
        # Neutralize their standardized contribution to the reference model.
        z[:,~measured]=0.
    model=bundle['classifier'];prob=model.predict_proba(z)
    scores=np.column_stack([x[:,ids].mean(1) for ids in bundle['marker_indices']])
    broad,confidence,margin,support=decisions(prob,model.classes_,scores,bundle['marker_floors'],c['annotation_min_probability'],c['annotation_min_margin'])
    final=broad.copy();mask=broad=='Epithelial';final[mask]='Epithelial_unresolved'
    if mask.any() and bundle['epithelial'] is not None:
        p=bundle['epithelial'].predict_proba(z[mask]);high=p.max(1)>=.7
        epi_labels=bundle['epithelial'].classes_[p.argmax(1)];ids=np.flatnonzero(mask);final[ids[high]]=epi_labels[high]
    result=pd.DataFrame({'cell_type':final,'cell_lineage':broad,'annotation_confidence':confidence,'annotation_margin':margin,
        'marker_supported':support,'annotation_status':np.where(broad=='Unknown','low_confidence_or_marker_unsupported','reference_supported')},index=index)
    for j,t in enumerate(model.classes_):result['prob_'+t]=prob[:,j]
    return result
