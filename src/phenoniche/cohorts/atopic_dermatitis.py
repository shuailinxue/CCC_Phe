"""Observed AD pair, audited outcome-blind label completion, shared CCC/GNN."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd,scipy.sparse as sp,anndata as ad,h5py,torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,f1_score,classification_report
from .colitis import tissue_components

PAPER='https://www.nature.com/articles/s41590-025-02267-8'
AUTHOR_URL='https://storage.googleapis.com/haniffalab/skin-fibroblast/adata_webportal_xenium.h5ad'
RAW_ROOT='https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/214/S-BIAD2214/Files/xenium/'
RAW_FOLDERS={'BK18':'output-XETG00272__0027131__Region_6__20240628__124459',
 'BK25':'output-XETG00155__0043604__BK25-SKI-27-FO-2-4-S3__20241010__133127',
 'BK30':'output-XETG00155__0043607__BK30-SKI-27-FO-1-2-S3__20241010__133127'}
IMMUNE=('B_Plasma','Lymphoid','Myeloid','Mast')


def ensure_downloads(root,donors=('BK18','BK25')):
    import requests
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    files=[('author_annotated.h5ad',AUTHOR_URL)]
    for donor in donors:
        folder=RAW_FOLDERS[donor]
        files.extend((f'xenium/{folder}/{name}',RAW_ROOT+folder+'/'+name) for name in ('cell_feature_matrix.h5','cells.csv.gz'))
    manifest=[]
    for name,url in files:
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():
            response=requests.get(url,stream=True,timeout=(20,120));response.raise_for_status();temporary=path.with_name(path.name+'.incomplete')
            with temporary.open('wb') as f:
                for block in response.iter_content(2**20):f.write(block)
            if response.headers.get('Content-Length') and temporary.stat().st_size!=int(response.headers['Content-Length']):raise IOError('Truncated download: '+url)
            temporary.replace(path)
        manifest.append({'path':str(path),'url':url,'bytes':path.stat().st_size})
    (root/'source_manifest.json').write_text(json.dumps({'paper':PAPER,'archive':'https://doi.org/10.6019/S-BIAD2214','files':manifest},indent=2))


def read_observed_sections(root,donors,counts_source='verified_raw'):
    """Exact author count/annotation alignment; independent archive audit.

    Some archive runs use different segmentation IDs from the published object.
    Never merge them by nearest coordinates or fabricate barcode correspondences.
    ``published_counts`` explicitly uses the author's integer counts layer, and
    records the failed independent raw-run match instead of silently joining it.
    """
    if counts_source not in ('verified_raw','published_counts'):raise ValueError('Invalid count source')
    root=Path(root);published=ad.read_h5ad(root/'author_annotated.h5ad',backed='r');blocks=[];audits=[]
    genes=published.var_names.str.upper()
    for donor in donors:
        obs=published.obs.loc[published.obs.DonorID.astype(str).eq(donor)].copy()
        obs.index=obs.cell_id.astype(str);folder=root/'xenium'/RAW_FOLDERS[donor]
        with h5py.File(folder/'cell_feature_matrix.h5') as f:
            m=f['matrix'];barcodes=pd.Index(m['barcodes'][:].astype(str));names=pd.Index(m['features']['name'][:].astype(str)).str.upper()
            raw=sp.csc_matrix((m['data'][:],m['indices'][:],m['indptr'][:]),shape=tuple(m['shape'][:])).T.tocsr()
        index=barcodes.get_indexer(obs.index);gi=names.get_indexer(genes)
        source_rows=np.flatnonzero(published.obs.DonorID.astype(str).to_numpy()==donor)
        author_counts=published.layers['counts'][source_rows].tocsr().astype(np.float32)
        if not np.isfinite(author_counts.data).all() or (author_counts.data<0).any() or not np.allclose(author_counts.data,np.round(author_counts.data)):
            raise ValueError('Published counts layer is not finite nonnegative integer counts')
        if (gi<0).any():raise ValueError('Published/raw gene panel mismatch')
        verified=False
        if (index>=0).all():
            xraw=raw[index][:,gi].astype(np.float32);difference=xraw-author_counts
            coords=pd.read_csv(folder/'cells.csv.gz').set_index('cell_id').loc[obs.index]
            verified=not difference.nnz and np.allclose(coords[['x_centroid','y_centroid']],obs[['x_centroid','y_centroid']],atol=1e-3)
        if counts_source=='verified_raw' and not verified:raise ValueError('Published annotation/raw identifiers do not align')
        x=author_counts;xy=obs[['x_centroid','y_centroid']].to_numpy(np.float32)
        if not np.allclose(xy,published.obsm['spatial'][source_rows],atol=1e-3):raise ValueError('Published centroid/obsm disagreement')
        obs['raw_barcode']=obs.index;obs['cell_id']=donor+'::'+obs.index;obs.index=obs.cell_id
        obs['patient']=donor;obs['image']=donor;obs['condition']='NL' if donor!='BK25' else 'AD'
        obs['library_counts']=np.asarray(x.sum(1)).ravel()
        b=ad.AnnData(x,obs=obs,var=pd.DataFrame(index=genes));b.obsm['spatial']=xy;blocks.append(b)
        audits.append({'section':donor,'condition':obs.condition.iloc[0],'cells':len(obs),'genes':len(genes),
            'median_counts':float(np.median(obs.library_counts)),'independent_archive_counts_verified':bool(verified),
            'raw_barcode_overlap':float((index>=0).mean()),'counts_source':counts_source,
            'counts_annotation_alignment':'Same published AnnData rows, named counts layer; never log X or imputed expression',
            'coordinate_unit':'original micrometers','original_site_status':obs.Site_status_binary.unique().tolist()})
    published.file.close();a=ad.concat(blocks,join='inner',merge='same');a.uns['source_audit_json']=json.dumps(audits)
    if not a.obs_names.is_unique:raise ValueError('Nonunique donor/barcode cell identifiers')
    return a


def complete_labels(a,annotation_donor_path,out,seed=52173):
    """Preserve author's labels; complete generic Other and VE/Pericyte using RNA.

    The annotation donor is never a separate ST anchor, never supplies GNN inputs
    or CCC values. Neither clinical condition nor spatial location is a classifier
    feature. Source whole-donor holdout tests audit this label completion.
    """
    out=Path(out);source=ad.read_h5ad(annotation_donor_path)
    common=source.var_names.intersection(a.var_names);source=source[:,common].copy()
    rng=np.random.default_rng(seed);label=source.obs.sp_annotation.astype(str).to_numpy();classes=np.unique(label)
    def expression(matrix):
        total=np.asarray(matrix.sum(1)).ravel();return np.log1p(matrix.toarray()*10000/np.maximum(total[:,None],1))
    sampled=[]
    for t in classes:
        rows=np.flatnonzero(label==t);sampled.extend(rng.choice(rows,min(1200,len(rows)),replace=False))
    sampled=np.sort(sampled);x=expression(source.X[sampled]);y=label[sampled]
    means=np.stack([x[y==t].mean(0) for t in classes]);variance=means.var(0)
    genes=np.argsort(-variance)[:700];x=x[:,genes]
    patient=source.obs.patient_id_paper.astype(str).to_numpy()[sampled]
    held=np.isin(patient,['patient_H8','patient_DM21']);scaler=StandardScaler().fit(x[~held]);clf=LogisticRegression(C=.1,class_weight='balanced',max_iter=700)
    clf.fit(scaler.transform(x[~held]),y[~held]);pred=clf.predict(scaler.transform(x[held]))
    report=classification_report(y[held],pred,output_dict=True,zero_division=0);pd.DataFrame(report).T.to_csv(out/'annotation_source_holdout.csv')
    audit={'outcome_labels_used':False,'spatial_coordinates_used':False,'annotation_donor_path':str(annotation_donor_path),
        'holdout':'Whole published donors H8 and DM21 excluded from audit classifier fit',
        'holdout_accuracy':float(accuracy_score(y[held],pred)),'holdout_macro_F1':float(f1_score(y[held],pred,average='macro')),
        'target_condition_used':False,'preserve_author_fibroblast_and_immune_annotations':True,'ambiguous_label':'Unresolved',
        'classifier_genes':common[genes].tolist(),'posterior_cutoff':.55,'scope':'Only cell annotation completion; donor expression never enters GNN/CCC targets'}
    if audit['holdout_accuracy']<.85:raise ValueError('Outcome-blind label completion fails whole-donor audit')
    scaler=StandardScaler().fit(x);clf.fit(scaler.transform(x),y)
    target=expression(a.X[:,a.var_names.get_indexer(common)])[:,genes];prob=clf.predict_proba(scaler.transform(target))
    predicted=clf.classes_[prob.argmax(1)];confidence=prob.max(1)
    coarse=np.full(len(a),'Unresolved',object);method=np.full(len(a),'RNA completion',object)
    fine=a.obs.finer_celltype.astype(str).to_numpy();broad=a.obs.broad_celltype.astype(str).to_numpy()
    mappings={'T':'Lymphoid','MigDC':'Myeloid','Mast cell':'Mast','Bcell/plasma cell':'B_Plasma','Schwann':'Schwann_cells'}
    for i,(f,b) in enumerate(zip(fine,broad)):
        known='Fibroblasts' if f.startswith('F') and ':' in f else mappings.get(f)
        if known:
            coarse[i]=known;confidence[i]=1.;method[i]='Published observed annotation';continue
        allowed=np.ones(len(clf.classes_),bool)
        if b=='Immune cell':allowed=np.isin(clf.classes_,IMMUNE)
        elif f=='VE/Pericyte':allowed=np.isin(clf.classes_,['Endothelial','Pericytes','Smooth_muscle','Lymphatic'])
        q=prob[i]*allowed
        if q.max()>=.55:coarse[i]=clf.classes_[q.argmax()];confidence[i]=q.max()
    a.obs['cell_type']=coarse;a.obs['cell_type_fine']=fine;a.obs['annotation_method']=method;a.obs['annotation_confidence']=confidence
    pd.crosstab(a.obs.image,a.obs.cell_type).to_csv(out/'annotation_section_counts.csv')
    audit['target_unresolved_fraction']=float(np.mean(coarse=='Unresolved'));audit['preserved_annotation_cells']=int(np.sum(method=='Published observed annotation'))
    if audit['target_unresolved_fraction']>.25:raise ValueError('Too many unresolved cell labels for credible directed CCC')
    (out/'annotation_audit.json').write_text(json.dumps(audit,indent=2));return a


def prepare_tissue(c,out):
    out=Path(out);cache=out/'input';cache.mkdir(parents=True,exist_ok=True)
    if (cache/'tissue.h5ad').exists():return ad.read_h5ad(cache/'tissue.h5ad')
    ensure_downloads(c['data_dir'],c['sections']);a=read_observed_sections(c['data_dir'],c['sections'],c['counts_source'])
    (out/'source_audit.json').write_text(a.uns['source_audit_json'])
    a=complete_labels(a,c['annotation_donor_counts'],out,c['seed'])
    count=np.asarray(a.X.sum(1)).ravel();a=a[(count>=10)&(np.diff(a.X.indptr)>=5)].copy()
    core,keep=tissue_components(a.obsm['spatial'],a.obs.image.to_numpy(),100.,100);a=a[keep].copy();a.obs['core']=core[keep]
    a.obs['fold']=np.zeros(len(a),np.int8);a.uns['patient_split']={str(v):0 for v in a.obs.core.unique()}
    a.uns['coordinate_units']='Original Xenium centroids in micrometers';a.write_h5ad(cache/'tissue.h5ad',compression='gzip')
    return a
