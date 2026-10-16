"""Observed skin pair adapter. Reuses production CCC, GNN and niche discovery."""
from pathlib import Path
from functools import partial
import hashlib,json,subprocess
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.io import mmread
from scipy.spatial import cKDTree
from scipy.spatial.distance import jensenshannon
import anndata as ad
import torch
from . import colitis as shared
from .case_novelty import find_case_niches
from ..gnn.config import GNNConfig
from ..gnn.training import fit,infer
from ..gnn.losses import ccc_loss
from ..gnn.evaluation import discover
from ..gnn.communication_metric import communication_embedding
from ..gnn.spatial_refinement import spatial_cluster_features
from ..storage import STORAGE_ROOT

ROOT=Path(__file__).resolve().parents[3]
MARKERS=('CXCL10','IL1B','CCL2','CCL5','CXCL9','CXCL11','IFNB1','IFNG','TNF','ISG15','MX1')


def assemble_export(root):
    """Read count matrices and cell IDs independently of serialized Seurat classes."""
    root=Path(root);target=root/'all_sections.h5ad'
    if target.exists():return ad.read_h5ad(target)
    export=root/'export'
    if not (export/'r_object_slots.json').exists():
        source=root/'main_object_sp_data.RDS'
        if not source.exists():raise FileNotFoundError('Downloaded original spatial object required: '+str(source))
        subprocess.run(['/home/xueshuailin/miniconda3/envs/cccphe/bin/Rscript',str(ROOT/'scripts/export_dermatomyositis.R'),str(source),str(export)],check=True)
    meta=pd.read_csv(export/'metadata.csv',index_col=False,low_memory=False)
    meta=meta.set_index('cell_id',drop=False);meta.index=meta.index.astype(str)
    if not meta.index.is_unique:raise ValueError('Duplicate annotation cell IDs')
    paths=sorted(export.glob('counts_*.mtx'))
    if not paths:raise FileNotFoundError('No original count matrices exported')
    blocks=[]
    for p in paths:
        genes=pd.Index(p.with_suffix('.genes.txt').read_text().splitlines()).str.upper()
        cells=p.with_suffix('.cells.txt').read_text().splitlines()
        m=mmread(p).T.tocsr().astype(np.float32)
        if m.shape!=(len(cells),len(genes)):raise ValueError('Count/export shape mismatch')
        # Negative-control probe features are never expression genes.
        valid=~genes.str.contains('NEGCONTROL|BLANK|UNASSIGNED|ANTISENSE',case=False,regex=True)
        b=ad.AnnData(m[:,valid],obs=pd.DataFrame(index=cells),var=pd.DataFrame(index=genes[valid]));blocks.append(b)
    a=ad.concat(blocks,join='outer',merge='same') if len(blocks)>1 else blocks[0]
    if not a.obs_names.is_unique or not a.var_names.is_unique:raise ValueError('Overlapping count layers or duplicate genes')
    order=meta.index.get_indexer(a.obs_names)
    if (order<0).any():raise ValueError('Counts missing published annotations')
    a.obs=meta.iloc[order].copy();a.obs.index=a.obs_names
    xy=pd.read_csv(export/'coordinates.csv');xy.cell_id=xy.cell_id.astype(str)
    if xy.cell_id.duplicated().any():raise ValueError('Duplicate coordinate cell IDs')
    xy=xy.set_index('cell_id');order=xy.index.get_indexer(a.obs_names)
    if (order<0).any():raise ValueError('Image coordinates not aligned to exact cell IDs')
    a.obsm['spatial']=xy.iloc[order][['x','y']].to_numpy(np.float32)
    a.obs['image']=xy.iloc[order].image.astype(str).to_numpy()
    # Keep only identifiers/annotations/QC, avoiding mixed-object R metadata.
    for col in a.obs:
        if a.obs[col].dtype=='object':a.obs[col]=a.obs[col].fillna('').astype(str)
    a.uns['counts_source']='Published original integer counts; no imputation or expression reconstruction'
    if not np.isfinite(a.X.data).all() or (a.X.data<0).any() or not np.allclose(a.X.data,np.round(a.X.data)):
        raise ValueError('Export did not contain original counts')
    temporary=target.with_name('all_sections.incomplete.h5ad')
    a.write_h5ad(temporary,compression='gzip');temporary.replace(target);return a


def column(frame, choices, description):
    for name in choices:
        if name in frame:return name
    raise ValueError(f'Cannot identify {description}; actual columns: {list(frame)}')


def audit_sections(a):
    """Quality and independent, pre-model inflammatory architecture summaries."""
    obs=a.obs
    patient=column(obs,('patient_id_paper','patient_id','patient','Patient','donor_id'),'patient')
    labels=column(obs,('sp_annotation','annotation','cell_type','sp_type','type'),'coarse cell type')
    obs['patient']=obs[patient].astype(str);obs['cell_type']=obs[labels].astype(str)
    obs['cell_type_fine']=obs[column(obs,('sp_type','type','sp_annotation',labels),'fine cell type')].astype(str)
    # Published IDs independently checked against supplementary patient metadata.
    obs['condition']=np.where(obs.patient.str.match(r'^patient_H\d+$'),'HC',
                              np.where(obs.patient.str.match(r'^patient_DM\d+$'),'DM','CLE'))
    obs['cell_id']=obs.index.astype(str)
    counts=np.asarray(a.X.sum(1)).ravel();genes=np.diff(a.X.indptr)
    totals=np.maximum(counts,1);marker_ids=a.var_names.get_indexer(MARKERS)
    present=marker_ids>=0
    signal=np.log1p(a.X[:,marker_ids[present]].toarray()*10000/totals[:,None]) if present.any() else np.zeros((len(a),1))
    inflammation=signal.mean(1);obs['inflammation_score']=inflammation
    xy=np.asarray(a.obsm['spatial']);records=[]
    types=sorted(obs.cell_type.unique());profiles={}
    for (patient,image),ids in obs.groupby(['patient','image'],observed=True).indices.items():
        ids=np.asarray(ids);v=xy[ids];dist,nn=cKDTree(v).query(v,k=min(30,len(ids)))
        local=inflammation[ids][nn].mean(1)
        profile=np.array([(obs.cell_type.iloc[ids]==t).mean() for t in types]);profiles[(patient,image)]=profile
        rec={'patient':patient,'image':image,'condition':str(obs.condition.iloc[ids[0]]),'cells':len(ids),
             'median_counts':float(np.median(counts[ids])),'median_genes':float(np.median(genes[ids])),
             'low_count_fraction':float((counts[ids]<20).mean()),'median_KNN30_radius_um':float(np.median(dist[:,-1])),
             'inflammation_local_p90':float(np.quantile(local,.9)),
             'inflammation_local_p90_p50':float(np.quantile(local,.9)-np.median(local)),
             'marker_panel_covered':int(present.sum()),'cell_type_entropy':float(-(profile*np.log(np.maximum(profile,1e-10))).sum())}
        records.append(rec)
    table=pd.DataFrame(records)
    # Quality eligibility is fixed before inspecting GNN results or disease-niche scores.
    table['quality_eligible']=(table.cells>=1000)&(table.median_counts>=50)&(table.median_genes>=20)&(table.low_count_fraction<=.2)
    table['quality_score']=(np.log1p(table.median_counts)+np.log1p(table.median_genes)+
                            .25*np.log1p(table.cells)-2*table.low_count_fraction)
    healthy=table[(table.condition=='HC')&table.quality_eligible]
    if healthy.empty:raise ValueError('No healthy section passes predeclared quality criteria')
    health=healthy.sort_values('quality_score',ascending=False).iloc[0]
    hp=profiles[(health.patient,health.image)]
    table['composition_JS_vs_selected_HC']=[float(jensenshannon(profiles[(r.patient,r.image)],hp)**2) for r in table.itertuples()]
    disease=table[(table.condition=='DM')&table.quality_eligible].copy()
    if disease.empty:raise ValueError('No DM section passes predeclared quality criteria')
    # Within the better-quality half, choose pronounced observed local inflammatory
    # architecture and composition divergence. This is an illustrative case, not
    # an unbiased group effect estimate or a selection on learned CCC/GNN accuracy.
    good=disease[disease.quality_score>=disease.quality_score.median()].copy()
    good['selection_score']=(good.inflammation_local_p90.rank(pct=True)+
                             good.inflammation_local_p90_p50.rank(pct=True)+
                             good.composition_JS_vs_selected_HC.rank(pct=True))
    case=good.sort_values(['selection_score','quality_score'],ascending=False).iloc[0]
    selected=[{'patient':str(r.patient),'image':str(r.image),'condition':str(r.condition)} for r in (health,case)]
    table['selected']=[any(r.patient==s['patient'] and r.image==s['image'] for s in selected) for r in table.itertuples()]
    return table,selected


def prepare_pair(c,out):
    root=Path(c['data_dir']);a=assemble_export(root)
    audit,selected=audit_sections(a)
    if c.get('selected_sections'):
        selected=c['selected_sections']
        if len(selected)!=2 or {s['condition'] for s in selected}!={'HC','DM'}:
            raise ValueError('Exactly one healthy and one DM section required')
        for s in selected:
            row=audit[(audit.patient==s['patient'])&(audit.image==s['image'])]
            if len(row)!=1 or row.iloc[0].condition!=s['condition'] or not bool(row.iloc[0].quality_eligible):
                raise ValueError('Requested section not uniquely identified or fails original QC: '+str(s))
        audit['selected']=[any(r.patient==s['patient'] and r.image==s['image'] for s in selected) for r in audit.itertuples()]
        # Recompute the composition comparison against the actually selected HC,
        # rather than retaining the initial automatic recommendation's HC.
        types=sorted(a.obs.cell_type.unique());profiles={}
        for (patient,image),ids in a.obs.groupby(['patient','image'],observed=True).indices.items():
            profiles[(patient,image)]=a.obs.cell_type.iloc[ids].value_counts(normalize=True).reindex(types,fill_value=0).to_numpy()
        hc=next(s for s in selected if s['condition']=='HC');hp=profiles[(hc['patient'],hc['image'])]
        audit['composition_JS_vs_selected_HC']=[float(jensenshannon(profiles[(r.patient,r.image)],hp)**2) for r in audit.itertuples()]
    clinical=Path(c['data_dir'])/'published_sample_metadata.csv'
    if clinical.exists():
        metadata=pd.read_csv(clinical).drop_duplicates('patient_id').rename(columns={'patient_id':'patient'})
        audit=audit.merge(metadata[['patient','site','sex','age','cancer','source']],on='patient',how='left',validate='many_to_one')
    audit.to_csv(out/'section_quality_and_selection.csv',index=False)
    (out/'selected_pair.json').write_text(json.dumps({'sections':selected,
        'rule':c.get('selection_rule','HC: best predeclared QC; DM: better-quality half, ranked local inflammation and composition divergence; no GNN/niche outcomes used'),
        'illustrative_selection':True,'markers':MARKERS,'source':'10.1038/s41467-026-74871-7',
        'coordinate_units':'Original Xenium centroid coordinates, micrometers; no pixel rescaling'},indent=2))
    keep=np.zeros(len(a),bool)
    for s in selected:keep|=(a.obs.patient==s['patient'])&(a.obs.image==s['image'])
    a=a[keep].copy();del keep
    count=np.asarray(a.X.sum(1)).ravel();detected=np.diff(a.X.indptr)
    a=a[(count>=10)&(detected>=5)&a.obs.cell_type.str.lower().ne('unassigned')].copy()
    core,keep=shared.tissue_components(a.obsm['spatial'],a.obs.image.astype(str).to_numpy(),c['core_link_um'],c['minimum_core_cells'])
    a=a[keep].copy();a.obs['core']=core[keep];a.obs['fold']=np.zeros(len(a),np.int8)
    a.uns['patient_split']={str(k):0 for k in a.obs.core.unique()}
    a.uns['coordinate_units']='micrometers; original Xenium centroids'
    # The common builder performs the exact audited directed physical-pair CCC.
    cache=out/'input';cache.mkdir(exist_ok=True);a.write_h5ad(cache/'tissue.h5ad',compression='gzip')
    return shared.prepare(c,out,tissue=a)


def summarize(a,data,z,h,labels,out,k):
    frame=a.obs.reset_index(drop=True).copy();frame['niche']=labels+1
    frame['x']=data.coordinates[:,0];frame['y']=data.coordinates[:,1]
    frame.to_csv(out/'assignments.csv.gz',index=False)
    freq=pd.crosstab(frame.condition,frame.niche,normalize='index').reindex(index=['HC','DM'],columns=np.arange(1,k+1),fill_value=0)
    freq.to_csv(out/'condition_niche_fractions.csv')
    means=np.stack([z[labels==j].mean(0) for j in range(k)])
    pd.DataFrame(means,index=np.arange(1,k+1),columns=np.arange(1,z.shape[1]+1)).to_csv(out/'niche_program_activity.csv',index_label='niche')
    np.save(out/'niche_composition.npy',np.stack([data.composition_target[labels==j].mean(0) for j in range(k)]))
    top=[]
    for j in range(k):
        for rank,f in enumerate(np.argsort(-(means[j]@h))[:10],1):
            r=data.features.iloc[f];top.append({'niche':j+1,'rank':rank,'sender':r.sender,'receiver':r.receiver,
                'LR':r.ligand+'–'+r.receptor,'mean_reconstructed_CCC':float(means[j]@h[:,f]),
                'main_program':int(np.argmax(means[j]*h[:,f]))+1})
    pd.DataFrame(top).to_csv(out/'niche_top10_CCC.csv',index=False)
    return frame


def clustering_values(data,arrays,h,c):
    if c.get('clustering_representation')!='ccc_profile_activity':
        return shared.clustering_representation(data,arrays,h,c)
    values,projection,audit=communication_embedding(arrays['P'],arrays['activity'],h,data.feature_scale,data.rows[0])
    ownership=[data.rows[0],np.empty(0,np.int64),np.empty(0,np.int64)]
    values=spatial_cluster_features(values,data.members,data.weights,ownership,c['spatial_mixing'])
    return values,None


def export_case_comparison(data,arrays,h,c,frame,out):
    # Calibrate on unpooled program representations: otherwise the optional
    # neighborhood smoothing could reintroduce shared support after block purge.
    values,_=clustering_values(data,arrays,h,{**c,'spatial_mixing':0.})
    novel=out/'case_novelty';novel.mkdir(exist_ok=True);z=np.asarray(arrays['Z'])
    scores,table,selection,audit=find_case_niches(values,data.composition_target,frame,data.members,data.weights,
        seed=c['seed'],block_um=c['novelty_block_um'],quantile=c['novelty_quantile'],
        min_cells=c['novelty_min_cells'],spatial_support=c['novelty_spatial_support'])
    audit['representation']=('Unpooled dictionary-aware relative CCC profile and weak log activity' if c.get('clustering_representation')=='ccc_profile_activity' else 'Unpooled sqrt(P) and weak log activity')
    scores.to_csv(novel/'cell_scores.csv.gz',index=False);table.to_csv(novel/'candidate_niches.csv',index=False)
    selection.to_csv(novel/'K_selection.csv',index=False);(novel/'audit.json').write_text(json.dumps(audit,indent=2))
    ntop=[]
    for label in sorted(scores.candidate_niche.unique()):
        if label==0:continue
        center=z[scores.candidate_niche.to_numpy()==label].mean(0);profile=center@h
        for rank,j in enumerate(np.argsort(-profile)[:10],1):
            f=data.features.iloc[j];ntop.append({'candidate_niche':f'R{label}','rank':rank,'sender':f.sender,
                'receiver':f.receiver,'LR':f.ligand+'–'+f.receptor,'mean_reconstructed_CCC':float(profile[j])})
    pd.DataFrame(ntop,columns=['candidate_niche','rank','sender','receiver','LR','mean_reconstructed_CCC']).to_csv(novel/'top10_CCC.csv',index=False)
    return table


def refresh_case_comparison(path):
    """Recompute only the optional downstream comparison; no GPU or training."""
    c=shared.read_config(path);out=Path(c['output_dir']);_,data=shared.load_prepared(c,out)
    arrays={key:np.load(out/'gnn'/f'{key}.npy',mmap_mode='r') for key in ('Z','P','activity')}
    h=np.load(out/'gnn/H.npy');frame=pd.read_csv(out/'assignments.csv.gz')
    table=export_case_comparison(data,arrays,h,c,frame,out)
    summary=json.loads((out/'summary.json').read_text());summary['case_novelty_candidates']=len(table)
    (out/'summary.json').write_text(json.dumps(summary,indent=2));return table


def run(path):
    torch.set_num_threads(8);c=shared.read_config(path);out=Path(c['output_dir']);out.mkdir(parents=True,exist_ok=True)
    signature=hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest();complete=out/'complete.json'
    if complete.exists():
        if json.loads(complete.read_text())['config_signature']!=signature:raise ValueError('Completed config differs; refusing overwrite')
        required=['summary.json','gnn/Z.npy','gnn/H.npy','assignments.csv.gz','case_novelty/cell_scores.csv.gz','niche_top10_CCC.csv','ccc_reconstruction.npy']
        if all((out/p).exists() for p in required):return out
    if (out/'cohort_config.json').exists() and json.loads((out/'cohort_config.json').read_text())!=c:
        raise ValueError('Existing run configuration differs; refusing to mix results')
    (out/'cohort_config.json').write_text(json.dumps(c,indent=2))
    if (out/'input/preprocessing.npz').exists():a,data=shared.load_prepared(c,out)
    else:a,data=prepare_pair(c,out)
    g=GNNConfig(seed=c['seed'],programs=32,hidden=c['hidden'],encoder_genes=len(data.encoder_indices),
        architecture='expression_gnn_residual_v2',message_layers=c['message_layers'],batch_size=c['batch_size'],
        max_epochs=c['epochs'],patience=c['patience'],device=c['device'],lambda_comp=0.,lambda_reg=.005,
        dictionary_initialization='nmf',output_dir=str(out/'gnn'),k_min=4,k_max=15,
        clustering_sample_size=12000,silhouette_sample_size=1200,near_optimal_score_tolerance=0.)
    if (out/'gnn/checkpoint.pt').exists():model=shared.restore_checkpoint(data,g)
    else:model=fit(data,g,rescale_activity=True,ccc_objective=partial(ccc_loss,kind='huber',delta=3.))
    if getattr(data,'_resident',None) is None:data.prepare_device(next(model.parameters()).device,out/'gnn/cache')
    arrays=infer(model,data,out/'gnn',c['batch_size']);z=np.asarray(arrays['Z']);h=np.load(out/'gnn/H.npy')
    values,projection=clustering_values(data,arrays,h,c)
    k,labels,_=discover(data,arrays,h,g,program_projection=projection,clustering_values=values,full_kmeans=True)
    pred=np.lib.format.open_memmap(out/'ccc_reconstruction.npy',mode='w+',dtype=np.float32,shape=data.targets.shape)
    for start in range(0,len(z),1024):pred[start:start+1024]=z[start:start+1024]@h
    pred.flush();del pred
    frame=summarize(a,data,z,h,labels,out,k)
    table=export_case_comparison(data,arrays,h,c,frame,out)
    summary={**data.audit,'selected_K':k,'programs':32,'conditions':['HC','DM'],'selected_sections':json.loads((out/'selected_pair.json').read_text())['sections'],
        'loss':'Feature-scaled CCC Huber(delta=3) + 0.005 program regularization',
        'case_novelty_candidates':len(table),'training_config':g.to_dict(),'group_replicates':1,
        'clustering_representation':c.get('clustering_representation','program_mixture_activity'),
        'interpretation':'Illustrative pair selected on independent QC/observed architecture; no group-level statistical inference'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2));complete.write_text(json.dumps({'config_signature':signature},indent=2))
    print('ANALYSIS COMPLETE',out,flush=True);return out


def recluster_saved(path,representation='ccc_profile_activity'):
    """Explicitly revise downstream distance only, retaining the initial control.

    Configuration changes are limited to clustering_representation. Original
    checkpoint bytes, CCC targets, Z/H, feature mask and training stay identical.
    """
    import shutil
    c=shared.read_config(path);out=Path(c['output_dir'])
    if not (out/'complete.json').exists():raise RuntimeError('Finish the original run before distance comparison')
    _,data=shared.load_prepared(c,out)
    control=out/'clustering_controls'/c['clustering_representation'];control.mkdir(parents=True,exist_ok=True)
    names=['assignments.csv.gz','condition_niche_fractions.csv','niche_program_activity.csv','niche_composition.npy',
           'niche_top10_CCC.csv','summary.json','gnn/K_selection.csv','gnn/niche_model.pt','gnn/selected_model/niche_labels.npy']
    for name in names:
        source=out/name;destination=control/name;destination.parent.mkdir(parents=True,exist_ok=True)
        if source.exists() and not destination.exists():shutil.copy2(source,destination)
    arrays={key:np.load(out/'gnn'/f'{key}.npy',mmap_mode='r') for key in ('Z','P','activity')};h=np.load(out/'gnn/H.npy')
    a=ad.read_h5ad(out/'input/tissue.h5ad');g=GNNConfig.from_json(out/'gnn/config.json')
    c={**c,'clustering_representation':representation}
    checkpoint=out/'gnn/checkpoint.pt';before=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    values,projection=clustering_values(data,arrays,h,c)
    # A partial reclustering must never be mistaken for a completed old result.
    (out/'complete.json').unlink()
    k,labels,_=discover(data,arrays,h,g,program_projection=projection,clustering_values=values,full_kmeans=True)
    frame=summarize(a,data,np.asarray(arrays['Z']),h,labels,out,k)
    candidates=export_case_comparison(data,arrays,h,c,frame,out)
    summary=json.loads((out/'summary.json').read_text());summary.update(selected_K=k,
        case_novelty_candidates=len(candidates),clustering_representation=representation)
    if representation=='ccc_profile_activity':
        _,projection,audit=communication_embedding(arrays['P'],arrays['activity'],h,data.feature_scale,data.rows[0])
        (out/'clustering_metric.json').write_text(json.dumps(audit,indent=2));np.save(out/'CCC_profile_projection.npy',projection)
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==before
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    Path(path).write_text(json.dumps(c,indent=2)+'\n');(out/'cohort_config.json').write_text(json.dumps(c,indent=2)+'\n')
    signature=hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()
    (out/'complete.json').write_text(json.dumps({'config_signature':signature},indent=2))
    (out/'case_novelty/model_identity.json').write_text(json.dumps({'checkpoint_sha256':before,
        'additional_training_epochs':0,'same_model_for_both_comparisons':True},indent=2))
    print('Reclustered saved model without training:',out,'K=',k,flush=True)
    return out
