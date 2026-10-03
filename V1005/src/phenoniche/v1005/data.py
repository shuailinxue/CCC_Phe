from pathlib import Path
import json
import anndata as ad
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
REPO = ROOT.parent
DATA = Path('/data1/xueshuailin/CCC_Phe_Niche/data')
PATHS = {'hbc1': DATA/'Xenium_HBC1/processed/HBC1_Xenium_processed.h5ad',
         'xenium5k': DATA/'xenium_prime_5k_breast/processed/XeniumPrime5K_Breast_V1002_processed.h5ad'}
TAXONOMY = {
'Invasive_Tumor':'Epithelial','Prolif_Invasive_Tumor':'Epithelial','DCIS_1':'Epithelial','DCIS_2':'Epithelial','Epithelial':'Epithelial',
'Macrophages_1':'Macrophage','Macrophages_2':'Macrophage','Macrophage':'Macrophage',
'CD4+_T_Cells':'T cell','CD8+_T_Cells':'T cell','T cell':'T cell','B_Cells':'B cell','B cell':'B cell',
'Stromal':'Fibroblast','Fibroblast':'Fibroblast','Endothelial':'Endothelial',
'Perivascular-Like':'Pericyte','Pericyte':'Pericyte','Myoepi_ACTA2+':'Myoepithelial','Myoepi_KRT15+':'Myoepithelial','Myoepithelial':'Myoepithelial',
'Mast_Cells':'Mast cell','Mast cell':'Mast cell','IRF7+_DCs':'Dendritic','LAMP3+_DCs':'Dendritic',
'Unlabeled':'Unknown','Unassigned':'Unknown','Stromal_&_T_Cell_Hybrid':'Unknown','T_Cell_&_Tumor_Hybrid':'Unknown'}

def load_data(dataset):
    a=ad.read_h5ad(PATHS[dataset])
    labels=a.obs.cell_type_coarse.astype(str)
    unknown=set(labels)-set(TAXONOMY)
    if unknown: raise ValueError(f'Unmapped original labels: {unknown}')
    mapped=labels.map(TAXONOMY);types=sorted(mapped.unique())
    ids=np.array([types.index(x) for x in mapped],dtype=np.int64)
    coords=np.asarray(a.obsm['spatial'],dtype=np.float32)
    assert a.obs_names.is_unique and np.isfinite(coords).all()
    return a,coords,ids,types

def json_write(path,obj):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(obj,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n')

def input_audit(dataset):
    """Read-only count QC and complete assay-measurable LR manifest; no new filtering."""
    import h5py
    import pandas as pd
    from phenoniche.v1002.lr_atlas import load_lr_atlas
    with h5py.File(PATHS[dataset]) as f:
        x=f['X'];indptr=x['indptr'][:];row_nnz=np.diff(indptr);total=0.;nonnegative=True;integer=True
        for start in range(0,len(x['data']),1_000_000):
            values=x['data'][start:start+1_000_000]
            total+=float(values.sum(dtype=np.float64));nonnegative &= bool((values>=0).all());integer &= bool(np.equal(values,np.floor(values)).all())
        names=f['var'][f['var'].attrs['_index']].asstr()[:];panel={g.upper() for g in names}
        qc={'n_cells':len(indptr)-1,'n_genes':len(names),'nnz':len(x['data']),'nnz_per_cell_median':float(np.median(row_nnz)),
            'zero_cells':int((row_nnz==0).sum()),'counts_per_cell_mean':total/(len(indptr)-1),'nonnegative':nonnegative,'integer_valued':integer}
    atlas=load_lr_atlas(REPO/'V1002/data/commuspace_human_lr_atlas.tsv')
    rows=[r for r in atlas.interactions if all(g in panel for g in r.ligand_components+r.receptor_components)]
    out=ROOT/'outputs'/dataset;out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame([{'lr_index':i,'lr_id':r.lr_id,'ligand':r.ligand,'receptor':r.receptor} for i,r in enumerate(rows)]).to_csv(out/'measurable_lr.csv',index=False)
    json_write(out/'input_matrix_qc.json',qc)
    return qc
