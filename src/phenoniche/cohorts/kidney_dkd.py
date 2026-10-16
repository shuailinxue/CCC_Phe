"""Public Xenium kidney counts, exact row annotations, anatomical ROI with halo.

Source: DOI 10.1038/s41586-026-10363-4, Zenodo 19868428. Neither the
authors' niche labels nor clinical endpoints enter inputs or ROI selection.
"""
from pathlib import Path
import json
import numpy as np,pandas as pd,scipy.sparse as sp,anndata as ad,h5py
from scipy.spatial import cKDTree
from .remote_hdf5 import CachedRangeReader
from .colitis import tissue_components
from ..gnn.data import neighborhoods

URL='https://zenodo.org/api/records/19868428/files/spatial_adata_xenium_cosmx_zenodo.h5ad/content'
BYTES=8745431314


def extract_sections(root,sections):
    """Fetch only exact public HDF5 byte ranges; never download/impute a panel."""
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    pending=[]
    for sample in sections:
        path=root/f'source_{sample}.h5ad'
        valid=False
        if path.exists():
            try:
                a=ad.read_h5ad(path,backed='r')
                valid=a.n_obs>30 and a.n_vars>1000 and {'source_row','annotation_updated','cell_id'}<=set(a.obs) and 'spatial' in a.obsm
                a.file.close()
            except (OSError,ValueError,KeyError):pass
        if not valid:pending.append(sample)
    if not pending:return
    with CachedRangeReader(URL,BYTES,root/'http_cache',prefetch_blocks=4) as reader:
        with h5py.File(reader,'r') as f:
            def column(name,rows=None):
                z=f['obs'][name]
                if isinstance(z,h5py.Group):
                    cats=z['categories'].asstr()[:];codes=z['codes'][:] if rows is None else z['codes'][rows]
                    return np.where(codes>=0,cats[np.maximum(codes,0)],'Unknown')
                if z.dtype.kind in ('O','S','U'):return z.asstr()[:] if rows is None else z.asstr()[rows]
                return z[:] if rows is None else z[rows]
            ident=column('orig_ident');tech=column('tech');genes=pd.Index(f['var/_index'].asstr()[:]).str.upper()
            for sample in pending:
                rows=np.flatnonzero((ident.astype(str)==str(sample))&(np.char.lower(tech.astype(str))=='xenium'))
                if len(rows)<30:raise ValueError(f'No Xenium section {sample}; check exact public sample identifier')
                lo,hi=int(rows.min()),int(rows.max()+1)
                if not np.array_equal(rows,np.arange(lo,hi)):raise ValueError('Noncontiguous source; refuse an approximate row join')
                sl=slice(lo,hi);names=['annotation_updated','immune_cell_annotation_combined','Condition','Age','Sex','nCount_RNA','nFeature_RNA']
                print('Reading exact original section rows',sample,len(rows),flush=True)
                frame=pd.DataFrame({name:column(name,sl) for name in names})
                frame['source_row']=rows;frame['original_cell_id']=column('_index',sl)
                frame['patient']=str(sample);frame['image']=str(sample)
                frame.index=pd.Index(str(sample)+'::'+frame.original_cell_id.astype(str),name='cell_id');frame['cell_id']=frame.index
                group=f['layers/counts'];ptr=group['indptr'][lo:hi+1].astype(np.int64);start,stop=int(ptr[0]),int(ptr[-1]);ptr-=start
                print('Fetching count-layer ranges',sample,'nonzeros',stop-start,flush=True)
                value=group['data'][start:stop]
                if not np.isfinite(value).all() or (value<0).any() or not np.allclose(value,np.round(value)):raise ValueError('Expected original integer counts')
                matrix=sp.csr_matrix((value,group['indices'][start:stop],ptr),shape=(len(rows),len(genes)))
                frame['library_counts']=np.asarray(matrix.sum(1)).ravel()
                if not np.allclose(frame.library_counts,frame.nCount_RNA):raise ValueError('Original source RNA count totals disagree')
                a=ad.AnnData(matrix,obs=frame,var=pd.DataFrame(index=genes));a.obsm['spatial']=f['obsm/spatial'][sl]
                a.uns['source_url']=URL;a.uns['counts_source']='Published layers/counts, exact original row alignment; Xenium only'
                tmp=root/f'source_{sample}.incomplete.h5ad';a.write_h5ad(tmp,compression='gzip');tmp.replace(root/f'source_{sample}.h5ad')
                print('Extracted original Xenium section',sample,a.shape,flush=True)


def parent_labels(frame):
    fine=frame.annotation_updated.astype(str).to_numpy()
    parent=np.where(fine=='iPT','PT',np.where(fine=='iTAL','TAL',fine)).astype(object)
    immune=frame.immune_cell_annotation_combined.astype(str).to_numpy()
    use=fine=='Immune';parent[use]=['Immune_unknown' if v in ('Unknown','Unassigned','nan') else v for v in immune[use]]
    return parent


def choose_roi(a,budget,seed):
    """One pre-model cortical ROI, by cortical coverage/compactness only.

    No condition comparison, CCC, expression states, injury labels, Z or niche
    outcomes are used. The chosen ROI is not optimized for case discrimination.
    """
    n=len(a);budget=min(int(budget),n);xy=np.asarray(a.obsm['spatial']);tree=cKDTree(xy)
    rng=np.random.default_rng(seed);candidates=np.sort(rng.choice(n,min(80,n),replace=False));records=[];choices=[]
    for row in candidates:
        distance,ids=tree.query(xy[row],k=budget);types=a.obs.cell_type.to_numpy()[ids]
        pt=float(np.mean(types=='PT'));glom=float(np.mean(np.isin(types,['Podo','MC1','EC_glom','PEC'])))
        span=np.maximum(np.ptp(xy[ids],axis=0),1);compact=float(min(span)/max(span))
        # Prefer balanced cortical tissue, penalizing medullary-only/sparse ROI.
        score=min(pt,.45)+2*min(glom,.12)+.1*compact
        records.append({'source_center':int(row),'x':xy[row,0],'y':xy[row,1],'PT_fraction':pt,'glomerular_fraction':glom,'compactness':compact,'radius_um':float(distance[-1]),'score':score})
        choices.append(np.asarray(ids,int))
    best=int(np.argmax([r['score'] for r in records]))
    for i,record in enumerate(records):record['selected']=i==best
    return np.sort(choices[best]),pd.DataFrame(records)


def prepare_tissue(c,out):
    out=Path(out);cache=out/'input';cache.mkdir(parents=True,exist_ok=True);saved=cache/'tissue.h5ad'
    if saved.exists():return ad.read_h5ad(saved)
    root=Path(c['data_dir']);extract_sections(root,c['sections']);blocks=[];audits=[];graphs=[]
    for sample in c['sections']:
        a=ad.read_h5ad(root/f'source_{sample}.h5ad');a.obs['cell_type']=parent_labels(a.obs)
        a.obs['cell_type_fine']=a.obs.annotation_updated.astype(str)
        source_conditions=set(a.obs.Condition.astype(str).str.lower())
        if source_conditions<={'control','healthy'}:a.obs['condition']='HC'
        elif source_conditions=={'dkd'}:a.obs['condition']='DKD'
        else:raise ValueError(f'Unexpected original conditions for paired kidney run: {source_conditions}')
        counts=np.asarray(a.X.sum(1)).ravel();keep=(counts>=10)&(np.diff(a.X.indptr)>=5)
        a=a[keep].copy();core,keep=tissue_components(a.obsm['spatial'],a.obs.image.astype(str).to_numpy(),100.,100)
        a=a[keep].copy();a.obs['core']=core[keep]
        ids=np.arange(len(a));roi=np.ones(len(a),bool)
        condition=a.obs.condition.iloc[0]
        budget=c['healthy_roi_centers'] if condition=='HC' else c.get('case_roi_centers',len(a))
        if len(a)>budget:
            chosen,scan=choose_roi(a,budget,c['seed']);scan.to_csv(out/f'{condition}_ROI_candidates.csv',index=False)
            roi[:]=False;roi[chosen]=True
            # Full-section neighbors first, halo retained, no artificial ROI edges.
            global_members=np.empty((len(a),30),np.int64);global_weights=np.empty((len(a),30),np.float32)
            for co in a.obs.core.unique():
                local=np.flatnonzero(a.obs.core.to_numpy()==co);m,w=neighborhoods(np.asarray(a.obsm['spatial'][local],np.float32));global_members[local]=local[m];global_weights[local]=w
            ids=np.unique(np.r_[chosen,global_members[chosen].ravel()])
            overview=a.obs[['cell_id','cell_type','core']].copy();overview['x']=a.obsm['spatial'][:,0];overview['y']=a.obsm['spatial'][:,1];overview['selected_ROI']=roi
            overview.to_csv(out/f'{condition}_full_section_ROI.csv.gz',index=False)
            if condition=='HC':overview.to_csv(out/'healthy_full_section_ROI.csv.gz',index=False)
        a.obs['analysis_center']=roi;a=a[ids].copy()
        graph_m=np.empty((len(a),30),np.int64);graph_w=np.empty((len(a),30),np.float32)
        for co in a.obs.core.unique():
            local=np.flatnonzero(a.obs.core.to_numpy()==co);m,w=neighborhoods(np.asarray(a.obsm['spatial'][local],np.float32));graph_m[local]=local[m];graph_w[local]=w
        if not roi.all():
            inverse=np.full(len(roi),-1,int);inverse[ids]=np.arange(len(ids));reported=np.flatnonzero(a.obs.analysis_center.to_numpy())
            # Copy original membership, including exact choices between tied KNN
            # distances; rebuilding a tree on a crop can break ties differently.
            graph_m[reported]=inverse[global_members[ids[reported]]];graph_w[reported]=global_weights[ids[reported]]
            if (graph_m[reported]<0).any():raise AssertionError('Missing physical ROI halo cell')
            np.testing.assert_array_equal(ids[graph_m[reported]],global_members[ids[reported]])
        graphs.append((graph_m+sum(len(b) for b in blocks),graph_w))
        audits.append({'sample':sample,'condition':a.obs.condition.iloc[0],'reported_centers':int(a.obs.analysis_center.sum()),
            'physical_cells_with_halo':len(a),'median_integer_counts':float(np.median(a.obs.library_counts)),
            'source':'Exact Xenium rows; author observed labels; no source niche labels',
            'parent_type_mapping':'iPT→PT; iTAL→TAL; Immune split by observed immune subtype',
            'coordinate_units':'Original Xenium micrometers; no rescaling','ROI_selection':'Same cortical structure rule in both sections, prior to CCC/GNN; all original KNN30 members retained'})
        a.obs['fold']=np.zeros(len(a),np.int8);blocks.append(a)
    a=ad.concat(blocks,join='inner',merge='same');a.uns['patient_split']={str(v):0 for v in a.obs.core.unique()}
    a.uns['coordinate_units']='Original Xenium micrometers';a.uns['source_paper']='10.1038/s41586-026-10363-4'
    if not a.obs_names.is_unique:raise ValueError('Repeated exact source cell IDs')
    (out/'source_audit.json').write_text(json.dumps(audits,indent=2))
    np.save(cache/'members.npy',np.vstack([m for m,w in graphs]));np.save(cache/'weights.npy',np.vstack([w for m,w in graphs]))
    tmp=cache/'tissue.incomplete.h5ad';a.write_h5ad(tmp,compression='gzip');tmp.replace(saved);return a
