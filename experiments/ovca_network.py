"""Sample centers while retaining all original cells in each V1 window."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
from scipy.spatial import cKDTree
from phenoniche.v1 import networks, storage
from phenoniche.spatial import channels

ROOT = Path('/home/nas3/biod/xueshuailin/CCC_Phe')
OUT = ROOT/'results/spatialfusion_ovca/phenomap_v1'
ATLAS = Path('data/commuspace_human_lr_atlas.tsv')
OV = Path('/data1/xueshuailin/CCC_Phe/workspace/data/spatialfusion_ovca/processed_OVCA.h5ad')
PT = ROOT/'data/nsclc_merfish_chen2024/processed/Pt73.h5ad'


def spatial_sample(xyz, ids, cap, rng):
    tree = cKDTree(xyz[ids])
    blocked = np.zeros(len(ids),bool)
    chosen = []
    for i in rng.permutation(len(ids)):
        if not blocked[i]:
            chosen.append(i)
            blocked[tree.query_ball_point(xyz[ids[i]],100)] = True
    return np.sort(ids[rng.choice(chosen,min(cap,len(chosen)),replace=False)])


def build(source, label, shuffle):
    meta = dict(source=storage.digest(source),atlas=storage.digest(ATLAS), seed=73,
                null_seed=31415 if shuffle else None, window=50, separation=100,
                train_cap=2048,test_cap=512,heldout_fraction=.20, type_column=label,
                source_code={Path(p).name:storage.digest(p) for p in
                    [__file__,channels.__file__,networks.__file__,storage.__file__]},
                versions=storage.versions(['anndata','numpy','scipy','numba','pandas']))
    dest=OUT/('network-'+storage.key(meta))
    dest.mkdir(parents=True, exist_ok=True)
    with storage.lock(dest):
        if (dest/'manifest.json').is_file():
            print(dest,flush=True)
            return
        a=ad.read_h5ad(source)
        xyz=np.asarray(a.obsm['spatial'],dtype=float)
        names,types=np.unique(a.obs[label].to_numpy(dtype=str),return_inverse=True)
        ids=a.obs_names.to_numpy(dtype=str)
        assert a.obs_names.is_unique and a.var_names.is_unique and np.isfinite(xyz).all()
        x=sparse.csr_matrix(a.X)
        assert np.isfinite(x.data).all() and (x.data>=0).all() and (x.data%1==0).all()
        audit=dict(cells=a.n_obs,genes=a.n_vars,nonzero_counts=int(x.nnz),
                   coordinate_extent=np.ptp(xyz,axis=0).tolist(), types=a.obs[label].value_counts().to_dict())
        if source==OV:
            pixel=np.asarray(a.obsm['spatial_px'])
            audit['pixel_conversion_max_error']=float(np.max(np.abs(xyz-.2125*pixel)))
            audit['transcript_count_match']=float(np.mean(np.asarray(x.sum(1)).ravel()==a.obs.transcript_counts))
            assert audit['pixel_conversion_max_error']<1e-4
        genes=a.var_names.to_numpy(dtype=str)
        del a
        samples=np.repeat('section',len(ids))
        train,test=networks.split_region(xyz,samples,50,.2)
        rng=np.random.default_rng(73)
        centers=np.sort(np.r_[spatial_sample(xyz,np.flatnonzero(train),2048,rng),
                             spatial_sample(xyz,np.flatnonzero(test),512,rng)])
        tree=cKDTree(xyz)
        windows=tree.query_ball_point(xyz[centers],50)
        # Compute the original normalization against every training-reference cell.
        expr,left,right,lr,signaling,scales=networks.expression(x,genes,pd.read_csv(ATLAS,sep='\t'),train)
        del x
        if shuffle:
            rng=np.random.default_rng(31415)
            side=xyz[:,0]>=xyz[test,0].min()-50
            strata=pd.DataFrame(dict(type=types,side=side)).groupby(['type','side']).indices
            for g in strata.values():
                expr[g]=expr[rng.permutation(g)]
        active=sparse.csr_matrix(expr)
        radii,group=np.unique([dict(Contact=20,ECM=20,Secreted=100,Unknown=50,Ambiguous=50)[s]
                               for s in signaling],return_inverse=True)
        args=xyz,types,len(names),expr,left,right,radii**2,group,active.indptr,active.indices
        def obs(i):
            return networks.observe(np.asarray(windows[i],dtype=np.int32),*args)
        obs(0)
        with ThreadPoolExecutor(12) as pool:
            for start in range(0,len(centers),256):
                part=dest/f'part-{start:08d}.npz'
                if part.exists():
                    continue
                m,s,e,v=zip(*pool.map(obs,range(start,min(start+256,len(centers)))))
                storage.save_npz(part,ids=ids[centers[start:start+len(m)]],mass=m,support=s,
                                 ptr=np.r_[0,np.cumsum([len(a) for a in e])],edge=np.concatenate(e),stats=np.concatenate(v))
                print(dest.name,start,len(centers),flush=True)
        storage.save_npz(dest/'schema.npz',ids=ids[centers],xyz=xyz[centers],names=names,
                        samples=samples[centers],train=train[centers],test=test[centers],
                        genes=genes,scales=scales,lr=np.asarray(lr),signaling=signaling,group=group,radii=radii,window=50)
        (dest/'audit.json').write_text(json.dumps(audit,indent=2))
        (dest/'manifest.json').write_text(json.dumps(meta,indent=2))
    print(dest,flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('dataset',choices=['ovca','null','pt73'])
    a=p.parse_args()
    build(PT if a.dataset=='pt73' else OV,'type_lvl1' if a.dataset=='pt73' else 'major_celltype',a.dataset=='null')
