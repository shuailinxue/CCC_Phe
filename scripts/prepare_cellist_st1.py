"""Read author counts and metadata; keep native coordinates until units verified."""
from pathlib import Path
import json
import anndata as ad, numpy as np,pandas as pd
from scipy.io import mmread
from scipy import sparse
D=Path('/data1/xueshuailin/CCC_Phe/workspace/data/nsclc_zhencaiwei/processed/ST1')
def run():
 p=D/'ST1_counts.h5ad'
 if p.exists():
  a=ad.read_h5ad(p,backed='r');print('Existing',a.shape,flush=True);a.file.close();return p
 meta=pd.read_csv(D/'metadata.csv');genes=pd.Index((D/'genes.txt').read_text().splitlines());cells=pd.Index((D/'cells.txt').read_text().splitlines());assert len(meta)==len(cells) and np.array_equal(meta.cell_id.to_numpy(),cells.to_numpy()) and genes.is_unique and cells.is_unique
 x=mmread(D/'counts.mtx').transpose().tocsr();assert x.shape==(len(cells),len(genes)) and np.issubdtype(x.dtype,np.integer) and x.data.min()>=0
 totals=np.asarray(x.sum(1)).ravel();delta=np.abs(totals-meta.nCount_RNA.to_numpy());assert np.max(delta)<1e-5, 'counts disagree with Seurat nCount_RNA'
 valid=meta.spatial_coordinates_valid.to_numpy(dtype=bool);meta.index=cells;spatial=meta[['x','y']].to_numpy();a=ad.AnnData(X=x[valid],obs=meta.iloc[np.flatnonzero(valid)].copy(),var=pd.DataFrame(index=genes));a.obsm['spatial_native']=spatial[valid];a.uns['source']='Cellist ST1 author Seurat RNA counts; one invalid coordinate excluded';a.uns['coordinate_unit']='unknown; native RDS x/y, not yet confirmed converted to micrometers';a.write_h5ad(p,compression='lzf');(D/'preparation.json').write_text(json.dumps({'source_cells':len(cells),'usable_cells':int(valid.sum()),'excluded_invalid_coordinates':int((~valid).sum()),'genes':len(genes),'counts_match_nCount_RNA':True,'coordinates':'native RDS x,y, unit unverified'},indent=2));print('Prepared',a.shape,'nnz',a.X.nnz,'max count difference',delta.max(),flush=True);return p
if __name__=='__main__':run()
