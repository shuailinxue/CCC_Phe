"""Small real H&E segmentation + UNI test, additional to real spot training."""
from pathlib import Path
import json
import numpy as np,pandas as pd
from .features import segment,extract,read_features


def image_smoke(c,source,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    # Choose the measured tissue center using observed cell locations only.
    meta=pd.read_hdf(Path(source)/'histology_cell_multifeatures.h5','metadata')
    x,y=meta[['center_x','center_y']].median().to_numpy();half=512
    bounds=(int(x-half),int(y-half),int(x+half),int(y+half))
    path=out/'metadata.csv'
    cells=pd.read_csv(path,index_col=0) if path.exists() else segment(c,Path(source)/'tissue_image.jpg',path,bounds)
    feature=out/'features.h5'
    if not feature.exists():extract(c,Path(source)/'tissue_image.jpg',cells,feature,max_cells=8)
    m,local,context=read_features(feature)
    if len(m)<2 or local.shape[1]!=1024 or not np.isfinite(local).all() or not np.isfinite(context).all():raise ValueError('Real image segmentation/UNI smoke failed')
    (out/'smoke.json').write_text(json.dumps({'real_histology':str(Path(source)/'tissue_image.jpg'),'crop_original_pixels':bounds,
        'segmented_instances':len(cells),'UNI_tested_cells':len(m),'feature_dimension':local.shape[1],'finite_local_context':True,'GPU_physical_index':2},indent=2))
    print('Real H&E segmentation and UNI smoke:',len(cells),'instances,',len(m),'features tested',flush=True)
