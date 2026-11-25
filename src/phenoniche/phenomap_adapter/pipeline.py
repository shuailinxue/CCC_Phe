"""Resumable per-section weak supervision, halo inference and audited exports."""
from pathlib import Path
import gc,json,os,hashlib,time
import numpy as np,pandas as pd,h5py,torch
from scipy.spatial import cKDTree
from torch_geometric.loader import DataLoader
from .model import make_model,zero_offset_scale,graph,halo_ids,load_upstream
from .features import read_features,segment,extract
from .annotation import annotate
from .genes import header


def atomic_json(path,value):
    tmp=Path(str(path)+'.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


def read_counts(path,genes):
    names=header(path);lookup={g.upper():g for g in names}
    present=[g for g in genes if g in lookup]
    frame=pd.read_csv(path,index_col=0,usecols=['spots']+[lookup[g] for g in present]);frame.columns=frame.columns.str.upper()
    frame=frame.reindex(columns=genes,fill_value=0.)
    frame.attrs['source_gene_present']=[g in lookup for g in genes]
    if not frame.index.is_unique or not np.isfinite(frame.to_numpy()).all() or (frame.to_numpy()<0).any():raise ValueError('Invalid original spot counts')
    return frame


def train(c,local,context,metadata,locations,counts,out,smoke=False):
    from torch.utils.data import Subset
    load_upstream(c['phenomap_repo'])
    from step1.data.dataset import SpotGraphDataset
    xy=metadata[['center_x','center_y']].to_numpy();spots=locations[['x','y']].to_numpy()
    diameter=float(c['spot_diameter_px']);radius=diameter/2*c['radius_ratio']
    tree=cKDTree(xy);members=tree.query_ball_point(spots,diameter/2)
    valid=np.array([i for i,m in enumerate(members) if len(m)>=c['minimum_spot_cells']],int)
    if len(valid)<12:raise ValueError('Fewer than 12 spots with usable cell centroids')
    if smoke:valid=valid[:min(len(valid),c['smoke_spots'])]
    # Contiguous stripe holdout, outcome-blind; capture circles cannot share cells.
    order=np.argsort(spots[valid,0],kind='stable');nv=max(3,int(len(valid)*.2));vpos=order[:nv];tpos=order[nv:]
    if set(np.concatenate([members[valid[i]] for i in tpos]))&set(np.concatenate([members[valid[i]] for i in vpos])):
        raise ValueError('Training/validation capture circles overlap physical cells')
    values=counts.to_numpy(np.float32);scaled,scale,active=zero_offset_scale(values[valid[tpos]])
    labels=values[valid]/scale;graphs=[]
    for index in valid:
        ids=np.sort(members[index]);graphs.append(graph(local[ids],context[ids],xy[ids],radius,c['max_neighbors']))
    dataset=SpotGraphDataset(graphs,torch.tensor(labels));generator=torch.Generator().manual_seed(c['seed'])
    loader=DataLoader(Subset(dataset,tpos.tolist()),batch_size=c['spot_batch'],shuffle=True,generator=generator)
    validation=DataLoader(Subset(dataset,vpos.tolist()),batch_size=c['spot_batch'],shuffle=False)
    model=make_model(c,local.shape[1],len(counts.columns)).to('cuda:0');model.active_gene_mask.copy_(torch.tensor(active,device='cuda:0'))
    optimizer=torch.optim.Adam(model.parameters(),lr=c['learning_rate']);history=[];start=0;best=float('inf');bad=0
    last=out/'last.pt';bestpath=out/'model.pt'
    if last.exists():
        state=torch.load(last,map_location='cuda:0',weights_only=False);model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        start=state['epoch']+1;best=state['best'];bad=state['bad'];history=state['history']
        torch.set_rng_state(state['torch_rng'].cpu());torch.cuda.set_rng_state(state['cuda_rng'].cpu());generator.set_state(state['loader_rng'].cpu())
    epochs=c['smoke_epochs'] if smoke else c['epochs']
    for epoch in range(start,epochs):
        if bad>=c['patience']:break
        model.train();metrics=[]
        for g,y in loader:
            g=g.to('cuda:0');y=y.to('cuda:0');optimizer.zero_grad(set_to_none=True)
            loss,mse,pcc,sparse=model._aggregate_and_compute_loss(g,y)
            if not torch.isfinite(loss):raise FloatingPointError('Non-finite PhenoMap training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
            metrics.append((float(loss.detach()),float(mse.detach())))
        model.eval();vm=[]
        with torch.inference_mode():
            for g,y in validation:
                vals=model._aggregate_and_compute_loss(g.to('cuda:0'),y.to('cuda:0'));vm.append((float(vals[0]),float(vals[1])))
        row={'epoch':epoch+1,'train_loss':float(np.mean(metrics,axis=0)[0]),'train_MSE':float(np.mean(metrics,axis=0)[1]),
            'validation_loss':float(np.mean(vm,axis=0)[0]),'validation_MSE':float(np.mean(vm,axis=0)[1])};history.append(row)
        print(row,flush=True)
        if row['validation_loss']<best-1e-6:
            best=row['validation_loss'];bad=0
            tmp=out/'model.incomplete.pt';torch.save({'model':model.state_dict(),'scale':scale,'active':active,'epoch':epoch+1},tmp);tmp.replace(bestpath)
        else:bad+=1
        state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'best':best,'bad':bad,'history':history,
            'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'loader_rng':generator.get_state()}
        tmp=out/'last.incomplete.pt';torch.save(state,tmp);tmp.replace(last)
        pd.DataFrame(history).to_csv(out/'training.csv',index=False)
    state=torch.load(bestpath,map_location='cuda:0',weights_only=False);model.load_state_dict(state['model']);model.eval()
    np.save(out/'gene_scale.npy',scale)
    atomic_json(out/'training_audit.json',{'train_spots':len(tpos),'validation_spots':len(vpos),'validation':'Contiguous x-stripe holdout; no shared physical capture cells; not independent patients',
        'seed':c['seed'],'best_epoch':int(state['epoch']),'gene_scale':'counts / training-spot gene max; no subtraction or affine offset',
        'activation':'Nonnegative Softplus output, absent training genes exactly masked to zero','active_training_genes':int(active.sum()),
        'radius_px':radius,'halo_layers':c['layers'],'feature_dim':local.shape[1],'smoke_only':smoke})
    return model,scale,radius,valid


def predict(c,model,scale,radius,local,context,metadata,out,genes,smoke=False):
    xy=metadata[['center_x','center_y']].to_numpy();grid=np.floor(xy/c['inference_tile_px']).astype(int);keys=np.unique(grid,axis=0)
    if smoke:
        # Pick a genuine densely populated interior tile, never fabricate coordinates.
        _,inverse,count=np.unique(grid,axis=0,return_inverse=True,return_counts=True);keys=keys[np.argsort(-count)[:1]]
    selected=np.flatnonzero(np.any(np.all(grid[:,None,:]==keys[None,:,:],axis=2),axis=1)) if smoke else np.arange(len(xy))
    rowmap=np.full(len(xy),-1,int);rowmap[selected]=np.arange(len(selected));path=out/'cell_expression.h5';tmp=out/'cell_expression.incomplete.h5'
    if path.exists():
        with h5py.File(path,'r') as h:
            np.testing.assert_array_equal(h['source_rows'][:],selected)
            if h['expression'].shape!=(len(selected),len(genes)) or not h['completed_tiles'][:].all():raise ValueError('Incomplete final prediction cache')
        return path
    with h5py.File(tmp,'a') as h:
        if 'expression' not in h:
            h.create_dataset('expression',shape=(len(selected),len(genes)),dtype='float32',chunks=(min(256,len(selected)),len(genes)),compression='lzf')
            h.create_dataset('source_rows',data=selected);h.create_dataset('completed_tiles',data=np.zeros(len(keys),bool))
        np.testing.assert_array_equal(h['source_rows'][:],selected)
        for k,key in enumerate(keys):
            if h['completed_tiles'][k]:continue
            core=np.flatnonzero((grid==key).all(1));halo=halo_ids(xy,core,radius,c['layers']);g=graph(local[halo],context[halo],xy[halo],radius,c['max_neighbors']).to('cuda:0')
            with torch.inference_mode():prediction=model(g.x,g.x_context,g.edge_index).cpu().numpy()
            value=prediction[np.searchsorted(halo,core)]*scale
            if not np.isfinite(value).all() or (value<0).any():raise ValueError('Invalid nonnegative expression export')
            h['expression'][rowmap[core]]=value;h['completed_tiles'][k]=True;h.flush()
            print('Inference tile',k+1,'/',len(keys),'core cells',len(core),flush=True)
    tmp.replace(path);return path


def export(c,path,metadata,genes,bundle,out,counts):
    import anndata as ad
    # Stream corrected predictions and annotations into backed h5ad without CSV expansion.
    with h5py.File(path,'r') as h:
        rows=h['source_rows'][:];meta=metadata.iloc[rows].copy();meta.index=meta.index.astype(str);meta.index.name='cell_id';annotations=[]
        measured=np.asarray(counts.attrs.get('source_gene_present',np.ones(len(genes),bool)),bool)
        for start in range(0,len(rows),2048):annotations.append(annotate(h['expression'][start:start+2048],genes,bundle,meta.index[start:start+2048],c,measured_genes=measured))
        anno=pd.concat(annotations);obs=meta.join(anno);obs['sample']=out.name;obs['expression_origin']='PhenoMap_inferred';obs['coordinate_units']='image_pixels'
        anno.to_csv(out/'cell_annotations.csv.gz')
        observed_fraction=(counts.to_numpy()>0).mean(0)
        trained=torch.load(out/'model.pt',map_location='cpu',weights_only=False)['active']
        var=pd.DataFrame({'source_gene_present':measured,'source_spot_detection_fraction':observed_fraction,'available_for_training':trained,
            'ccc_gene_eligible_at_10pct':(observed_fraction>=.1)&trained},index=genes)
        # h5py direct dense array fill, using AnnData-created metadata encoding.
        stub=ad.AnnData(X=None,obs=obs,var=var,shape=(len(rows),len(genes)))
        stub.obsm['spatial']=meta[['center_x','center_y']].to_numpy(np.float32)
        stub.uns['provenance']={'expression':'Inferred continuous nonnegative expression, not measured single-cell UMI','scaling':'Zero-offset gene max; no per-cell spot minima','annotations':'Outcome-blind breast atlas; confidence and marker support retained','coordinates':'Pixels; µm calibration required before sigma=20µm CCC',
            'CCC_feature_mask':'Use observed source_spot_detection_fraction / ccc_gene_eligible_at_10pct; >0 of Softplus-imputed RNA is NOT an empirical detection rate'}
        final=out/'cells.h5ad';temp=out/'cells.incomplete.h5ad';stub.write_h5ad(temp)
        with h5py.File(temp,'a') as dest:
            if 'X' in dest:del dest['X']
            x=dest.create_dataset('X',shape=h['expression'].shape,dtype='float32',chunks=h['expression'].chunks,compression='lzf');x.attrs['encoding-type']='array';x.attrs['encoding-version']='0.2.0'
            for start in range(0,len(rows),2048):x[start:start+2048]=h['expression'][start:start+2048]
        temp.replace(final)
    atomic_json(out/'annotation_summary.json',{'cell_types':anno.cell_type.value_counts().to_dict(),'lineages':anno.cell_lineage.value_counts().to_dict(),
        'unknown_fraction':float(anno.cell_type.eq('Unknown').mean()),'reference_validation':bundle['validation'],
        'note':'No forced fractions and no claim that reference validation is single-cell ST ground truth'})
    return final


def run_slice(c,source,out,genes,bundle,smoke=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);source=Path(source)
    old=source/'histology_cell_multifeatures.h5';featurepath=out/'cell_features.h5';metapath=out/'cell_metadata.csv'
    if old.exists() and c['reuse_existing_features']:featurepath=old
    elif not featurepath.exists():
        metadata=pd.read_csv(metapath,index_col=0) if metapath.exists() else segment(c,source/'tissue_image.jpg',metapath)
        extract(c,source/'tissue_image.jpg',metadata,featurepath)
    metadata,local,context=read_features(featurepath)
    if (metadata.crop_size<c['minimum_crop_px']).any():
        keep=(metadata.crop_size>=c['minimum_crop_px']).to_numpy();metadata=metadata.iloc[np.flatnonzero(keep)];local=local[keep];context=context[keep]
    locations=pd.read_csv(source/'locs.csv',index_col=0);counts=read_counts(source/'cnts.csv',genes).reindex(locations.index)
    c=dict(c);diameter=source/'spot_diameter_pixel.txt'
    if diameter.exists():c['spot_diameter_px']=float(diameter.read_text())
    else:
        # This source cohort uses the same ST-array geometry. Infer pixel scale
        # from its regular spot lattice; never silently transplant 300px to another platform.
        spacing=np.median(cKDTree(locations[['x','y']]).query(locations[['x','y']],k=2)[0][:,1])
        c['spot_diameter_px']=float(spacing*c['cohort_spot_diameter_to_spacing'])
    atomic_json(out/'geometry.json',{'diameter_px':c['spot_diameter_px'],'source':'Existing per-slice diameter' if diameter.exists() else 'Same-cohort lattice ratio calibrated from TNBC1_CN1_C2',
        'units':'Image pixels; no invented micrometer scale','cell_features':str(featurepath)})
    model,scale,radius,valid=train(c,local,context,metadata,locations,counts,out,smoke)
    path=predict(c,model,scale,radius,local,context,metadata,out,genes,smoke);result=export(c,path,metadata,genes,bundle,out,counts)
    del model,local,context;gc.collect();torch.cuda.empty_cache();return result
