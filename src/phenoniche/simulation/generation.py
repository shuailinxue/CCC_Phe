"""Verified scCube APIs; self-contained seed reference and symmetric tissue layout."""
from pathlib import Path
import json,inspect,hashlib,importlib.metadata as metadata
import numpy as np,pandas as pd,anndata as ad
from ..ccc.lr_atlas import load_lr_atlas
from ..training import set_seed
from ..data import resolve_device
TYPES=('Tumor','Normal epithelial','Fibroblast','Macrophage','T cell')
SCCUBE_COMMIT='d5b423d296da17d0fd5b4d12e92262167a642445'
SPIDERNET_COMMIT='c142960620f2fa584e8843c9acf48d719c1f2f8a'

def check_environment():
    from scCube import scCube,SPatternGeneratorCustom
    required={'sc_adata','celltype_key','cell_key','target_num','epoch_num','used_device'}
    if not required<=set(inspect.signature(scCube.train_vae_and_generate_cell).parameters):raise RuntimeError('Unsupported scCube generation API')
    if 'prop_list' not in inspect.signature(SPatternGeneratorCustom.simulate_mixing).parameters:raise RuntimeError('Unsupported scCube spatial API')
    return {p:metadata.version(p) for p in ('scCube','numpy','pandas','anndata','scanpy','torch','POT','scikit-learn')}

def make_lr(config):
    atlas=load_lr_atlas(config.lr_atlas);rng=np.random.default_rng(config.seed)
    pool=[v for v in atlas.interactions if len(v.ligand_components)==len(v.receptor_components)==1 and v.ligand!=v.receptor]
    order=rng.permutation(len(pool));selected=[];used=set()
    for i in order:
        v=pool[i]
        if not ({v.ligand,v.receptor}&used):selected.append(v);used.update((v.ligand,v.receptor))
        if len(selected)==2*config.mi_lr_count+5*config.basal_lr_per_type:break
    if len(selected)<2*config.mi_lr_count+5*config.basal_lr_per_type:raise ValueError('Not enough gene-disjoint real LR pairs')
    keys={(v.ligand,v.receptor) for v in selected}
    for i in order:
        v=pool[i]
        if (v.ligand,v.receptor) not in keys:selected.append(v);keys.add((v.ligand,v.receptor))
        if len(selected)==config.n_lr:break
    table=pd.DataFrame([{'lr_id':v.lr_id,'ligand':v.ligand,'receptor':v.receptor,'mi':1 if i<config.mi_lr_count else 2 if i<2*config.mi_lr_count else 0,'loading':1/config.mi_lr_count if i<2*config.mi_lr_count else 0.,'source':'CommuSpace union; intended sender/receiver roles are simulation design'} for i,v in enumerate(selected)])
    table['sender_role']=config.sender_type;table['receiver_role']=config.receiver_type
    table['basal_type']=''
    for t,name in enumerate(TYPES):
        ids=np.arange(2*config.mi_lr_count+t*config.basal_lr_per_type,2*config.mi_lr_count+(t+1)*config.basal_lr_per_type)
        table.loc[ids,'basal_type']=name;table.loc[ids,['sender_role','receiver_role']]=name
    if config.mi_loadings is not None:
        for mi in (1,2):
            weights=np.asarray(config.mi_loadings[mi-1],float);table.loc[table.mi==mi,'loading']=weights/weights.sum()
    return table

def seed_reference(config,lr):
    rng=np.random.default_rng(config.seed);genes=sorted(set(lr.ligand)|set(lr.receptor))
    genes+=['SYN_MARKER_'+str(i) for i in range(25)]
    genes+=['SYN_BG_'+str(i) for i in range(config.n_genes-len(genes))]
    if len(genes)>config.n_genes:raise ValueError('n_genes insufficient for actual selected LR union and type markers')
    n=config.reference_cells_per_type;types=np.repeat(TYPES,n);means=np.full((len(types),len(genes)),.3)
    markers=[genes.index('SYN_MARKER_'+str(i)) for i in range(25)]
    for i in range(5):means[i*n:(i+1)*n,np.asarray(markers[i*5:(i+1)*5])]=12.
    # Type-specific real LR genes, not just synthetic markers, define basal
    # communication. Roles/means are explicit synthetic assumptions, not
    # experimentally curated cell-type expression priors. No region labels.
    for i,name in enumerate(TYPES):
        for row in lr[lr.basal_type==name].itertuples():
            positions=[genes.index(row.ligand),genes.index(row.receptor)]
            means[:,positions]=config.basal_lr_offtype_mean
            means[i*n:(i+1)*n,np.asarray(positions)]=config.basal_lr_mean
    counts=rng.negative_binomial(3,3/(3+means)).astype(np.float32)
    obs=pd.DataFrame({'Cell_type':types,'Cell':['seed_'+str(i) for i in range(len(types))]},index=['seed_'+str(i) for i in range(len(types))])
    return ad.AnnData(counts,obs=obs,var=pd.DataFrame(index=genes))


def calibrate_type_means(rates,labels,reference):
    """Explicit count-space calibration; uses seed types, never region/MI labels.

    Preserve scCube's within-type relative variation while restoring seed
    gene means. This is a hybrid simulator, not a claim of native VAE fidelity.
    """
    result=np.asarray(rates,np.float64).copy()
    for name in TYPES:
        ids=np.flatnonzero(np.asarray(labels)==name)
        if not len(ids):continue
        seed=np.asarray(reference.X[reference.obs.Cell_type.to_numpy()==name],np.float64)
        target=seed.mean(0);mean=result[ids].mean(0)
        ratio=np.divide(target,mean,out=np.zeros_like(target),where=mean>1e-12)
        block=result[ids]*ratio
        # A fully dead decoded gene has no relative variation to retain.
        block[:,mean<=1e-12]=target[mean<=1e-12]
        result[ids]=block
    return result


def marker_audit(counts,labels,genes):
    records=[]
    for index,name in enumerate(TYPES):
        ids=np.flatnonzero(np.asarray(labels)==name)
        if not len(ids):continue
        own=[list(genes).index(f'SYN_MARKER_{j}') for j in range(5*index,5*index+5)]
        others=[list(genes).index(f'SYN_MARKER_{j}') for j in range(25) if j//5!=index]
        records.append({'cell_type':name,'own_marker_mean':float(counts[ids][:,own].mean()),'other_marker_mean':float(counts[ids][:,others].mean())})
    return records


def basal_lr_audit(counts,labels,genes,lr):
    records=[]
    for name in TYPES:
        links=lr[lr.basal_type==name]
        if links.empty:continue
        positions=[list(genes).index(g) for g in sorted(set(links.ligand)|set(links.receptor))]
        own=np.asarray(labels)==name
        records.append({'cell_type':name,'LR_pairs':len(links),'own_LR_gene_mean':float(counts[own][:,positions].mean()),'other_type_LR_gene_mean':float(counts[~own][:,positions].mean())})
    return records

def generate(config,output):
    from scCube import scCube,SPatternGeneratorCustom
    set_seed(config.seed);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    lr=make_lr(config);ref=seed_reference(config,lr);ref.write_h5ad(output/'seed_reference.h5ad')
    rng=np.random.default_rng(config.seed);halves=[];type_arrays=[];sample_arrays=[]
    half_total=config.n_cells//2;counts=np.full(config.replicates,half_total//config.replicates);counts[:half_total%config.replicates]+=1
    for sample,n in enumerate(counts):
        # Use scCube's verified Poisson spatial generator, then a documented
        # area-preserving half-disc transform and bilateral paired control.
        gen=SPatternGeneratorCustom(ref,cell_num=int(n),celltype_key='Cell_type',set_seed=True,seed=config.seed+sample,spatial_size=1,select_celltype=list(TYPES))
        pattern=gen.simulate_mixing(prop_list=[1,0,0,0,0])
        u=np.asarray(pattern.point_x);theta=np.pi*(np.asarray(pattern.point_y)-.5)
        radius=config.outer_radius_um*np.sqrt(np.maximum(u,1e-9))
        xy=np.column_stack((radius*np.cos(theta),radius*np.sin(theta)))
        types=np.empty(n,dtype=int)
        for region,(keep,prop) in enumerate([(radius<config.core_radius_um,config.core_proportions),((radius>=config.core_radius_um)&(radius<config.core_radius_um+config.edge_width_um),config.edge_proportions),(radius>=config.core_radius_um+config.edge_width_um,config.exterior_proportions)]):
            ids=np.flatnonzero(keep);types[ids]=rng.choice(5,len(ids),p=prop)
            if len(ids)>=5 and region==1:types[ids[:5]]=np.arange(5)
        halves.append(xy);type_arrays.append(types);sample_arrays.append(np.full(n,sample))
    half_xy=np.concatenate(halves);half_types=np.concatenate(type_arrays);half_samples=np.concatenate(sample_arrays)
    target={name:int((half_types==i).sum()) for i,name in enumerate(TYPES)}
    # AnnData/scCube preprocessing mutates its input buffer in place. Give
    # it an owned copy so the count-space calibration reference stays counts.
    cube=scCube();normalized=cube.pre_process(pd.DataFrame(ref.X.T.copy(),index=ref.var_names,columns=ref.obs['Cell']),ref.obs.copy(),is_normalized=False)
    checkpoint=output/'sccube_generator.pth'
    if checkpoint.is_file():
        meta,data=cube.load_vae_and_generate_cell(sc_adata=normalized,celltype_key='Cell_type',cell_key='Cell',target_num=target,hidden_size=config.sc_cube_hidden,load_path=str(checkpoint),used_device=resolve_device(config.device))
    else:
        meta,data=cube.train_vae_and_generate_cell(sc_adata=normalized,celltype_key='Cell_type',cell_key='Cell',target_num=target,batch_size=min(128,len(ref)),epoch_num=config.sc_cube_epochs,lr=config.sc_cube_lr,hidden_size=config.sc_cube_hidden,save_model=True,save_path=str(output),project_name='sccube_generator',used_device=resolve_device(config.device))
    if not meta.Cell.is_unique or set(data.columns)!=set(meta.Cell):raise ValueError('scCube output IDs do not align')
    values=data.loc[ref.var_names,meta.Cell].to_numpy(np.float32).T
    if not np.isfinite(values).all():raise ValueError('scCube generated nonfinite expression')
    reference_max=max(float(normalized.X.max()),1.)
    # VAE generates log-normalized expression. Invert and sample counts;
    # clipping to the seed dynamic range is explicit and recorded.
    rates=np.expm1(np.clip(values,0,reference_max))
    raw_counts=rng.poisson(rates).astype(np.float32)
    np.save(output/'sccube_uncalibrated_half_counts.npy',raw_counts)
    generated_labels=meta.Cell_type.to_numpy()
    calibrated=calibrate_type_means(rates,generated_labels,ref) if config.expression_calibration=='type_mean' else rates
    counts_generated=rng.poisson(calibrated).astype(np.float32) if config.expression_calibration=='type_mean' else raw_counts
    fidelity={'mode':config.expression_calibration,'raw_scCube':marker_audit(raw_counts,generated_labels,ref.var_names),'final_counts':marker_audit(counts_generated,generated_labels,ref.var_names),'policy':'Seed type means only; no niche/MI labels, no CCC modification'}
    fidelity['basal_LR_profiles']=basal_lr_audit(counts_generated,generated_labels,ref.var_names,lr)
    (output/'expression_fidelity.json').write_text(json.dumps(fidelity,indent=2)+'\n')
    half_expr=np.empty((half_total,len(ref.var_names)),np.float32)
    for i,name in enumerate(TYPES):
        dest=np.flatnonzero(half_types==i);source=np.flatnonzero(meta.Cell_type.to_numpy()==name)
        if len(dest)!=len(source):raise ValueError('scCube output cell-type target counts mismatch')
        half_expr[dest]=counts_generated[source]
    mirrored=half_xy.copy();mirrored[:,0]*=-1
    xy=np.concatenate((mirrored,half_xy));types=np.tile(half_types,2);samples=np.tile(half_samples,2)
    local_xy=xy.copy();xy[:,0]+=samples*(2*config.outer_radius_um+500)
    radius=np.linalg.norm(local_xy,axis=1);edge=(radius>=config.core_radius_um)&(radius<config.core_radius_um+config.edge_width_um)
    truth=np.where(radius<config.core_radius_um,0,np.where(~edge,1,np.where(local_xy[:,0]<0,2,3)))
    obs=pd.DataFrame({'cell_id':['sim_'+str(i) for i in range(config.n_cells)],'cell_type':np.asarray(TYPES)[types],'sample_id':samples,'mirror_pair':np.tile(np.arange(half_total),2),'radius_um':radius,'local_x':local_xy[:,0],'local_y':local_xy[:,1],'true_niche':truth},index=['sim_'+str(i) for i in range(config.n_cells)])
    a=ad.AnnData(np.tile(half_expr,(2,1)),obs=obs,var=pd.DataFrame(index=ref.var_names));a.obsm['spatial']=xy.astype(np.float32)
    a.uns['simulation_provenance']={'scCube_commit':SCCUBE_COMMIT,'SpiderNet_simulation_commit':SPIDERNET_COMMIT,'seed_reference':'synthetic negative-binomial profiles, not patient measurements','expression_transform':f'invert clipped log1p values; {config.expression_calibration} seed-type mean calibration; Poisson sample; duplicate paired control before MI injection','geometry':'scCube Poisson points -> uniform half-disc -> bilateral reflection; µm','versions':check_environment()}
    a.uns['simulation_provenance']['basal_LR_profiles']='Real LR identities with synthetic type-specific basal expression; no anatomical/MI labels used for basal profiles'
    a.write_h5ad(output/'before_injection.incomplete.h5ad');(output/'before_injection.incomplete.h5ad').replace(output/'before_injection.h5ad');lr.to_csv(output/'lr.csv',index=False)
    return a,lr
