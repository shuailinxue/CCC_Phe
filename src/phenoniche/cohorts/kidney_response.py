"""Fixed-niche receiver response analysis; no encoder or CCC target mutation.

Evidence chain: original observed CCC → external ligand-target potential →
non-LR measured receiver RNA → sender exposure / response spatial association.
NicheNet potentials are unsigned, indirect and not receptor-specific. Spatial
blocks describe this tissue, never independent patient replication.
"""
from pathlib import Path
import hashlib,json,re
import numpy as np,pandas as pd,scipy.sparse as sp
from scipy.stats import rankdata,spearmanr
from sklearn.metrics import average_precision_score,roc_auc_score
from sklearn.decomposition import PCA
from .paired_skin import input_root
from .kidney_downstream import gene_effects,unique_physical_contacts,hallmark_sets

RAW_PRIOR=Path('/home/nas3/biod/xueshuailin/CCC_Phe/workspace/data/priors/raw/nichenet_alternate.rds')
PRIOR_URL='https://zenodo.org/records/7074291/files/ligand_target_matrix_nsga2r_final.rds'
PRIOR_MD5='b09606b04b2d4490418d9028c0e58b9f'
FOCUS=(('CD8+','CXCL14'),('Macro','APP'),('Macro','MIF'))


def is_immunoglobulin(gene):
    return bool(re.match(r'^(IGH[VDJ]|IG[KL][VDJC]|IGH[AG][0-9]|IGH[DEM]$)',str(gene)))


def ccc_genes(features):
    return {g.upper() for value in pd.concat([features.ligand,features.receptor]) for g in re.split(r'[&+_| ]+',str(value)) if g}


def target_weights(prior_column,genes,baseline_detection,exclude,top=30):
    """Panel projection of real potentials; selection never uses niche effects.

    NicheNet operates on the measured receiver background. Global-human top100
    coverage is recorded separately: it cannot be silently treated as 30 strong
    full-human targets when a sparse panel only measures a small intersection.
    """
    prior=prior_column[prior_column>0].sort_values(ascending=False,kind='stable')
    allowed={g for g,d in zip(genes,baseline_detection) if d>=.05 and g not in exclude}
    selected=prior[prior.index.isin(allowed)].head(top)
    return selected


def response_score(expression,genes,baseline_rows,targets):
    """Fixed-prior weighted standardized RNA; no outcome-based gene selection."""
    if len(targets)<5:return None
    ids=pd.Index(genes).get_indexer(targets.index)
    if (ids<0).any():raise ValueError('Target gene alignment failed')
    x=expression[:,ids].toarray();bg=x[baseline_rows]
    mean=bg.mean(0);sd=np.maximum(bg.std(0),.25)
    score=np.clip((x-mean)/sd,-5,5)@(targets.to_numpy()/targets.sum())
    score-=score[baseline_rows].mean()
    return score


def partial_rank(x,y,covariates):
    """Residual ranks after prespecified measured-state covariates."""
    design=np.column_stack([np.ones(len(x))]+[rankdata(covariates[:,i]) for i in range(covariates.shape[1])])
    design[:,1:]=(design[:,1:]-design[:,1:].mean(0))/np.maximum(design[:,1:].std(0),1e-8)
    rx=rankdata(x);ry=rankdata(y)
    return rx-design@np.linalg.lstsq(design,rx,rcond=None)[0],ry-design@np.linalg.lstsq(design,ry,rcond=None)[0]


def spatial_association(x,y,xy,covariates,seed=52173,bootstraps=499,block_um=160.):
    """No cell-level disease p: association + spatial-block bootstrap interval."""
    rx,ry=partial_rank(x,y,covariates)
    block=np.floor(xy/block_um).astype(int);_,labels=np.unique(block,axis=0,return_inverse=True)
    counts=np.bincount(labels);keep=counts>=3
    bx=(np.bincount(labels,weights=rx)/counts)[keep];by=(np.bincount(labels,weights=ry)/counts)[keep]
    def corr(a,b):
        return float(np.corrcoef(a,b)[0,1]) if min(np.std(a),np.std(b))>1e-10 else np.nan
    rho=corr(bx,by);rng=np.random.default_rng(seed);null=[]
    if len(bx)>=10:
        for _ in range(bootstraps):
            ids=rng.integers(0,len(bx),len(bx));null.append(corr(bx[ids],by[ids]))
    lo,hi=np.nanquantile(null,[.025,.975]) if null else (np.nan,np.nan)
    return {'block_correlation':rho,'block_CI_low':lo,'block_CI_high':hi,'blocks':len(bx),'cells':len(x)},rx,ry,labels


def sender_exposure(u,v,w,types,receiver_rows,sender_expression):
    """Receiver-independent exposure; no receptor or target RNA multiplication."""
    uu=np.r_[u,v];vv=np.r_[v,u];ww=np.r_[w,w];n=len(types)
    wanted=np.zeros(n,bool);wanted[receiver_rows]=True
    keep=wanted[vv]&(types[uu]=='TAL');uu,vv,ww=uu[keep],vv[keep],ww[keep]
    den=np.bincount(vv,weights=ww,minlength=n);out=np.zeros((n,sender_expression.shape[1]),np.float32)
    for j in range(out.shape[1]):
        out[:,j]=np.bincount(vv,weights=ww*sender_expression[uu,j],minlength=n)/np.maximum(den,1e-12)
    return out,den


def ensure_receiver_response(source):
    import anndata as ad,pyreadr,torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(8);threadpool_limits(8)
    source=Path(source);out=source/'downstream/receiver_response';out.mkdir(parents=True,exist_ok=True)
    if not RAW_PRIOR.exists():raise FileNotFoundError(f'Download official human NicheNet v2 matrix from {PRIOR_URL} to {RAW_PRIOR}')
    raw=RAW_PRIOR.read_bytes()
    if hashlib.md5(raw).hexdigest()!=PRIOR_MD5:raise ValueError('NicheNet v2 official checksum mismatch')
    signature={'code':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'prior_md5':PRIOR_MD5,
        'checkpoint':hashlib.sha256((source/'gnn/checkpoint.pt').read_bytes()).hexdigest(),
        'assignments':hashlib.sha256((source/'condition_niches/assignments.csv.gz').read_bytes()).hexdigest(),
        'upstream_analysis':hashlib.sha256((source/'downstream/complete.json').read_bytes()).hexdigest()}
    if (out/'complete.json').exists() and json.loads((out/'complete.json').read_text())==signature:return out
    cells=pd.read_csv(source/'condition_niches/assignments.csv.gz');cache=input_root(source)
    a=ad.read_h5ad(cache/'tissue.h5ad');order=pd.Index(a.obs.cell_id.astype(str)).get_indexer(cells.cell_id.astype(str))
    if (order<0).any() or len(np.unique(order))!=len(cells):raise ValueError('Exact receiver cell alignment failed')
    a=a[order].copy();genes=np.asarray(a.var_names);log=sp.load_npz(source/'downstream/observed_log_expression.npz')
    np.testing.assert_allclose(a.obsm['spatial'],cells[['x','y']].to_numpy(),atol=1e-3)
    features=pd.read_csv(cache/'features.csv');exclude=ccc_genes(features)
    exclude|={g for g in genes if is_immunoglobulin(g)}
    types=cells.cell_type.to_numpy();xy=cells[['x','y']].to_numpy()
    valid=cells.analysis_center.to_numpy(bool);case=valid&cells.condition.eq('DKD').to_numpy();r2=case&cells.label.eq('R2').to_numpy();shared=case&cells.label.str.startswith('B').to_numpy()
    prior=pyreadr.read_r(str(RAW_PRIOR))[None];prior.index=prior.index.astype(str).str.upper();prior.columns=prior.columns.astype(str).str.upper()
    np.save(out/'source_rows.npy',np.arange(len(cells)))
    pairs=pd.read_csv(source/'downstream/R2_same_sender_receiver_CCC.csv')
    candidate=features[(features.sender=='TAL')&features.receiver.isin(['CD8+','Macro'])]
    ligands=sorted(set(candidate.ligand)&set(prior.columns)&set(genes))
    print('Receiver response: authentic prior loaded;',len(ligands),'TAL ligands covered',flush=True)
    # All DKD physical pairs, independent of R2 selection, same original weights.
    members=np.load(cache/'members.npy');weights=np.load(cache/'weights.npy');eligible=np.flatnonzero(case)
    has_tal=(types[members[eligible]]=='TAL').any(1);has_receiver=np.isin(types[members[eligible]],['CD8+','Macro']).any(1)
    eligible=eligible[has_tal&has_receiver]
    u,v,w=unique_physical_contacts(eligible,members,weights,xy)
    receiver_rows=np.flatnonzero(case&np.isin(types,['CD8+','Macro']))
    exposure,opportunity=sender_exposure(u,v,w,types,receiver_rows,log[:,pd.Index(genes).get_indexer(ligands)].toarray())
    comp=np.load(cache/'composition.npy',mmap_mode='r');pc=np.zeros((len(cells),3));pc[case]=PCA(3,random_state=52173).fit_transform(np.asarray(comp[case]))
    total=np.asarray(a.X.sum(1)).ravel();detected=np.asarray((a.X>0).sum(1)).ravel()
    all_effects=[];coverage=[];activity=[];links=[];percell=[];associations=[];module_effects=[]
    sets,_=hallmark_sets(Path('/home/nas3/biod/xueshuailin/CCC_Phe/workspace/data/gene_sets'));module_scores=[]
    for receiver in ['CD8+','Macro']:
        take=np.flatnonzero(r2&(types==receiver));bg=np.flatnonzero(shared&(types==receiver));rows=np.flatnonzero(valid&(types==receiver))
        if min(len(take),len(bg))<30:raise ValueError(f'Insufficient same-type cells for {receiver}')
        effects=gene_effects(log,genes,take,bg);effects['receiver']=receiver;all_effects.append(effects)
        universe=effects[effects.universe&~effects.gene.isin(exclude)&effects.gene.isin(prior.index)].set_index('gene')
        selected=universe[(universe.log2FC>=.25)&(universe.niche_detection>=.1)].nlargest(100,'log2FC').index
        baseline_detection=np.asarray((log[bg]>0).mean(0)).ravel()
        for ligand in sorted(candidate.loc[candidate.receiver==receiver,'ligand'].unique()):
            if ligand not in prior.columns:
                coverage.append({'receiver':receiver,'ligand':ligand,'prior_covered':False,'retained_targets':0});continue
            targets=target_weights(prior[ligand],genes,baseline_detection,exclude)
            global100=set(prior[ligand].nlargest(100).index)
            coverage.append({'receiver':receiver,'ligand':ligand,'prior_covered':True,'retained_targets':len(targets),
                'retained_target_names':';'.join(targets.index),'global_top100_covered':len(set(targets.index)&global100),
                'target_policy':'top30 among detected nonLR measured background; no niche-effect selection'})
            p=prior[ligand].reindex(universe.index).fillna(0).to_numpy();label=universe.index.isin(selected)
            ap=average_precision_score(label,p) if label.any() else np.nan
            activity.append({'receiver':receiver,'ligand':ligand,'background_genes':len(label),'up_genes':int(label.sum()),
                'AUROC':roc_auc_score(label,p) if label.any() and not label.all() else np.nan,'AUPRC':ap,
                'AUPRC_minus_prevalence':ap-float(label.mean()),'prior_effect_spearman':float(spearmanr(p,universe.log2FC).statistic)})
            score=response_score(log,genes,bg,targets)
            if score is None:continue
            global_rank=prior[ligand].rank(method='min',ascending=False)
            for g,potential in targets.items():
                e=effects.loc[effects.gene==g].iloc[0]
                links.append({'receiver':receiver,'ligand':ligand,'target':g,'prior_potential':potential,'full_prior_rank':int(global_rank[g]),'log2FC':e.log2FC,
                    'niche_detection':e.niche_detection,'shared_detection':e.background_detection,'excluded_LR':False})
            for name,mask in [('Healthy',valid&cells.condition.eq('HC').to_numpy()),('DKD shared',shared),('DKD R2',r2)]:
                ids=np.flatnonzero(mask&(types==receiver));module_effects.append({'receiver':receiver,'ligand':ligand,'group':name,'cells':len(ids),'mean_target_score':float(score[ids].mean())})
            # Receptors recorded from original CCC; conservative AND-complex expression.
            receptor_names=sorted(set(candidate.loc[(candidate.receiver==receiver)&(candidate.ligand==ligand),'receptor']))
            receptor_genes=sorted({g for value in receptor_names for g in re.split(r'[&+_| ]+',value) if g in genes})
            rec=np.asarray(log[:,pd.Index(genes).get_indexer(receptor_genes)].mean(1)).ravel() if receptor_genes else np.zeros(len(cells))
            ids=np.flatnonzero(case&(types==receiver)&(opportunity>1e-5))
            cov=np.column_stack([np.log1p(total[ids]),np.log1p(detected[ids]),np.log1p(opportunity[ids]),rec[ids],pc[ids]])
            summary,rx,ry,blocks=spatial_association(exposure[ids,ligands.index(ligand)],score[ids],xy[ids],cov)
            summary.update({'receiver':receiver,'ligand':ligand,'receptors':';'.join(receptor_names),'target_genes':len(targets)});associations.append(summary)
            percell.extend({'source_row':int(row),'receiver':receiver,'ligand':ligand,'cell_id':cells.cell_id.iloc[row],
                'label':cells.label.iloc[row],'x':xy[row,0],'y':xy[row,1],'sender_exposure':float(exposure[row,ligands.index(ligand)]),
                'target_score':float(score[row]),'exposure_residual':float(rx[j]),'target_residual':float(ry[j]),'spatial_block':int(blocks[j])}
                for j,row in enumerate(ids))
        # Outcome-independent Hallmark scores, all measured non-LR genes at 5% baseline detection.
        allowed=set(genes[baseline_detection>=.05])-exclude
        for name,gs in sets.items():
            t=sorted(gs&allowed)
            if len(t)<8:continue
            score=response_score(log,genes,bg,pd.Series(np.ones(len(t)),index=t))
            groups={'Healthy':valid&cells.condition.eq('HC').to_numpy(),'DKD shared':shared,'DKD R2':r2}
            for group,mask in groups.items():
                ids=np.flatnonzero(mask&(types==receiver));module_scores.append({'receiver':receiver,'module':name,'group':group,
                    'cells':len(ids),'measured_nonLR_genes':len(t),'mean_score':float(score[ids].mean()),'genes':';'.join(t)})
    pd.concat(all_effects).to_csv(out/'receiver_gene_effects.csv.gz',index=False)
    for name,rows in [('prior_coverage',coverage),('ligand_activity',activity),('ligand_target_links',links),('response_by_group',module_effects),
                      ('spatial_associations',associations),('receiver_cells',percell),('hallmark_response_modules',module_scores)]:
        pd.DataFrame(rows).to_csv(out/(name+('.csv.gz' if name=='receiver_cells' else '.csv')),index=False)
    # Evidence for LR candidates, preserving controversial rather than invented mechanisms.
    evidence=[{'LR':'CXCL14–CXCR4','interpretation':'Candidate co-expression/proximity; literature reports conflicting modulation. Not asserted to activate CXCR4.',
        'sources':'https://pubmed.ncbi.nlm.nih.gov/25451233/;https://pubmed.ncbi.nlm.nih.gov/25372750/'},
        {'LR':'NicheNet ligand→target links','interpretation':'Unsigned integrated regulatory potential; indirect, not receptor-specific or direct causal validation.',
        'sources':'https://doi.org/10.1038/s41592-019-0667-5;https://zenodo.org/records/7074291'}]
    pd.DataFrame(evidence).to_csv(out/'mechanism_evidence.csv',index=False)
    audit={'prior':PRIOR_URL,'official_md5':PRIOR_MD5,'prior_shape':list(prior.shape),'target_policy':'Same-type DKD shared detection≥5%, exclude ALL CCC ligand/receptor and immunoglobulin genes; top30 by authentic prior among measured background; global-human prior ranks and top100 coverage recorded; ≥5 targets',
        'receiver_comparison':'Unique physical CD8+ / Macro cells, R2 versus same-type DKD shared B niches; healthy shown as context',
        'ligand_activity':'NicheNet-inspired AUPRC/AUROC on measured nonLR universe, not a literal reproduction of package assay',
        'spatial_exposure':'Original physical pairs over all DKD centers, deduplicated mean weights; TAL weighted mean observed log1pTP10K ligand RNA; no own receptor/target factor',
        'spatial_association':'Partial-rank residuals adjusted log library, log detected genes, TAL opportunity, receptor RNA and 3 local-composition PCs; means of 160µm blocks (≥3 cells); 499 block bootstrap intervals',
        'interpretation':'Within-section descriptive associations, not patient-level disease tests; no causal activation sign from unsigned ligand-target potential',
        'model_refit':False,'CCC_mutated':False,'excluded_CCC_genes':sorted(exclude),'focus':FOCUS}
    (out/'audit.json').write_text(json.dumps(audit,indent=2));(out/'complete.json').write_text(json.dumps(signature,indent=2));return out
