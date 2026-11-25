"""Tile-owned Cellpose instances and original PhenoMap UNI feature encoder."""
from pathlib import Path
import gc,json,sys
import numpy as np,pandas as pd,h5py,torch
from PIL import Image
from scipy.ndimage import find_objects,center_of_mass
from scipy.spatial import cKDTree
from .model import load_upstream


def cellpose_model(c):
    from cellpose.models import CellposeModel
    return CellposeModel(gpu=True,device=torch.device('cuda:0'),pretrained_model=c['cellpose_weights'])


def instance_centers(mask,bounds,next_id=1,minimum=8,maximum=256):
    x0,x1,y0,y1,ix0,ix1,iy0,iy1=bounds;labels=np.unique(mask);labels=labels[labels>0]
    centers=center_of_mass(mask>0,labels=mask,index=labels);objects=find_objects(mask)
    rows=[]
    for label,(y,x) in zip(labels,centers):
        by,bx=objects[int(label)-1];size=max(by.stop-by.start,bx.stop-bx.start);xx=x+x0;yy=y+y0
        # A centroid belongs to exactly one tile's core; retain whole-instance bbox.
        if ix0<=xx<ix1 and iy0<=yy<iy1 and minimum<=size<=maximum:
            rows.append((next_id,xx,yy,size));next_id+=1
    return rows,next_id


def segment(c,image_path,out,crop=None):
    load_upstream(c['phenomap_repo'])
    from preprocessing.segment_cells import iter_tiles,is_blank_tile
    Image.MAX_IMAGE_PIXELS=None
    with Image.open(image_path) as im:image=np.asarray(im.convert('RGB')) if crop is None else np.asarray(im.crop(crop).convert('RGB'))
    origin=np.zeros(2) if crop is None else np.asarray(crop[:2]);h,w=image.shape[:2]
    model=cellpose_model(c);rows=[];next_id=1
    tiles=Path(out).parent/'segmentation_tiles';tiles.mkdir(exist_ok=True)
    for number,b in enumerate(iter_tiles(h,w,c['segment_tile'],c['segment_overlap'])):
        tile_cache=tiles/f'{number:06d}.csv'
        if tile_cache.exists():
            cached=pd.read_csv(tile_cache).to_records(index=False).tolist();rows.extend(cached)
            if cached:next_id=max(next_id,max(int(v[0]) for v in cached)+1)
            continue
        x0,x1,y0,y1,*_=b;tile=image[y0:y1,x0:x1]
        if is_blank_tile(tile,245.,5.):new=[]
        else:
            mask,_,_=model.eval(tile,batch_size=8,flow_threshold=.4,cellprob_threshold=0.,min_size=15,invert=False)
            new,next_id=instance_centers(mask,b,next_id,c['minimum_crop_px'],256)
        rows.extend(new);tmp=tile_cache.with_suffix('.tmp');pd.DataFrame(new,columns=['cell_id','center_x','center_y','crop_size']).to_csv(tmp,index=False);tmp.replace(tile_cache)
        if number%20==0:print('Segmentation tiles',number+1,'instances',len(rows),flush=True)
    del model,image;gc.collect();torch.cuda.empty_cache()
    if not rows:raise ValueError('No cells detected by segmentation')
    frame=pd.DataFrame(rows,columns=['cell_id','center_x','center_y','crop_size'])
    frame[['center_x','center_y']]+=origin
    # Only near-identical tile-boundary centers are deduplicated, never arbitrary KNN cells.
    duplicate=cKDTree(frame[['center_x','center_y']]).query_pairs(2.)
    drop={max(i,j) for i,j in duplicate};frame=frame.drop(index=list(drop)).reset_index(drop=True)
    frame['cell_id']=np.arange(1,len(frame)+1);frame=frame.set_index('cell_id')
    frame.to_csv(out);return frame


def extract(c,image_path,metadata,path,max_cells=None):
    load_upstream(c['phenomap_repo'])
    from step1.models.extractors import UNIExtractor
    from timm.data import create_transform,resolve_data_config
    # Local weight reader from upstream: no hub login/download or repository writes.
    extractor=UNIExtractor.__new__(UNIExtractor)
    model=extractor._load_uni_from_local_cache().eval().to('cuda:0')
    transform=create_transform(**resolve_data_config(model.pretrained_cfg,model=model))
    Image.MAX_IMAGE_PIXELS=None
    with Image.open(image_path) as im:image=np.asarray(im.convert('RGB'))
    if max_cells is not None:metadata=metadata.iloc[:max_cells].copy()
    xy=metadata[['center_x','center_y']].to_numpy();sizes=metadata.crop_size.to_numpy()
    temporary=Path(str(path)+'.incomplete');temporary.parent.mkdir(parents=True,exist_ok=True)
    with h5py.File(temporary,'a') as h:
        if 'cell_ids' not in h:
            h.create_dataset('cell_ids',data=metadata.index.to_numpy(np.int64));h.create_dataset('coordinates',data=xy.astype(np.float32));h.create_dataset('crop_size',data=sizes)
        np.testing.assert_array_equal(h['cell_ids'][:],metadata.index)
        for name,pad in [('local',20),('context',100)]:
            dataset=h.get(name);written=int(dataset.attrs.get('rows_written',0)) if dataset is not None else 0
            for start in range(written,len(metadata),c['feature_batch']):
                batch=[]
                for (x,y),size in zip(xy[start:start+c['feature_batch']],sizes[start:start+c['feature_batch']]):
                    half=int(size)//2+pad;x=int(x);y=int(y)
                    patch=image[max(0,y-half):min(image.shape[0],y+half),max(0,x-half):min(image.shape[1],x+half)]
                    if patch.size==0:raise ValueError('Cell coordinate outside histology image')
                    batch.append(transform(Image.fromarray(patch)))
                with torch.inference_mode():value=model(torch.stack(batch).to('cuda:0')).float().cpu().numpy()
                if dataset is None:dataset=h.create_dataset(name,shape=(len(metadata),value.shape[1]),dtype='float16',chunks=(min(256,len(metadata)),value.shape[1]),compression='lzf')
                dataset[start:start+len(value)]=value;dataset.attrs['rows_written']=start+len(value);h.flush()
                if start%32768==0:print('UNI',name,start,'/',len(metadata),flush=True)
        h.attrs['protocol']='Original PhenoMap UNI pretrained encoder; RGB patches, local pad20/context pad100; no inferred RNA'
    temporary.replace(path);del model,image;gc.collect();torch.cuda.empty_cache();return path


def read_features(path):
    with h5py.File(path,'r') as h:custom='cell_ids' in h
    if custom:
        with h5py.File(path,'r') as h:
            meta=pd.DataFrame(h['coordinates'][:],columns=['center_x','center_y'],index=h['cell_ids'][:]);meta['crop_size']=h['crop_size'][:]
            return meta,h['local'][:].astype(np.float32),h['context'][:].astype(np.float32)
    meta=pd.read_hdf(path,'metadata');local=pd.read_hdf(path,'features');context=pd.read_hdf(path,'context_features')
    if not meta.index.is_unique:raise ValueError('Duplicate physical cell IDs')
    return meta,local.loc[meta.index].to_numpy(np.float32),context.loc[meta.index].to_numpy(np.float32)
