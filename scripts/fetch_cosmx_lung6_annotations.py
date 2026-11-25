"""Fetch only annotation fields, never published niche labels or embeddings."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import h5py,numpy as np,pandas as pd
from phenoniche.cohorts.remote_hdf5 import CachedRangeReader
ROOT=Path('/home/nas3/biod/xueshuailin/CCC_Phe/workspace/data/cosmx_lung6')

def value(obj):
 if isinstance(obj,h5py.Group):return value(obj['categories'])[obj['codes'][:]]
 return obj.asstr()[:] if h5py.check_string_dtype(obj.dtype) else obj[:]

if __name__=='__main__':
 with CachedRangeReader('https://ndownloader.figshare.com/files/46841842',2755776882,ROOT/'annotation_http_cache',block_size=1<<20) as reader,h5py.File(reader,'r') as f:
  sample=value(f['obs/sample']);keep=sample=='LUSC-6'
  assert keep.sum()>10000,'Missing author sample'
  table=pd.DataFrame({k:value(f['obs'][k])[keep] for k in ['sample','fov','cell_ID','cell_type']})
  table['source_index']=value(f['obs/_index'])[keep]
  table.to_csv(ROOT/'author_cell_annotations.csv.gz',index=False)
  (ROOT/'annotation_provenance.json').write_text(json.dumps({'source':'https://figshare.com/articles/dataset/CosMx_human_NSCLC/25976224','file':'cosmx_human_nsclc_clustered.h5ad','fields_used':['sample','fov','cell_ID','cell_type'],'published_niche_labels_read':False,'processed_embedding_used':False,'annotation_rows':len(table),'http_bytes_fetched':reader.fetched_bytes},indent=2)+'\n')
  print(table.head().to_string(),flush=True);print(table.cell_type.value_counts(),flush=True)
