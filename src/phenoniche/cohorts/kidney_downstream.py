"""Post-fit kidney niche interpretation: spatial contacts, LR and cell states.

No encoder fitting, clinical endpoint, author niche label, or patient-level test.
Cell expression is observed RNA, never decoded or imputed. Original CCC stays
unchanged. Repeated physical edges are deduplicated ONLY for contact analysis.
"""
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd,scipy.sparse as sp,torch,requests
from scipy.stats import hypergeom
from sklearn.decomposition import PCA
from statsmodels.stats.multitest import multipletests
from ..gnn.model import physical_weights
from .paired_skin import input_root

HALLMARK_URL='https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=MSigDB_Hallmark_2020'
CONTEXTS=(('R1','B'),('R2','TAL'),('R2','Macro'),('R3','PC'),('R4','PT'))


def hallmark_sets(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);path=root/'MSigDB_Hallmark_2020.gmt'
    if not path.exists():
        response=requests.get(HALLMARK_URL,timeout=60);response.raise_for_status()
        lines=response.text.splitlines()
        if len(lines)!=50 or any(len(line.split('\t'))<4 for line in lines):raise ValueError('Invalid public Hallmark resource')
        tmp=path.with_suffix('.incomplete');tmp.write_text(response.text);tmp.replace(path)
    result={}
    for line in path.read_text().splitlines():
        fields=line.split('\t');result[fields[0]]=set(g.upper() for g in fields[2:] if g)
    if len(result)!=50:raise ValueError('Expected the real 50 Hallmark sets')
    (root/'MSigDB_Hallmark_2020.source.json').write_text(json.dumps({'source':HALLMARK_URL,'original_collection':'MSigDB Hallmark',
        'library_version':'2020','provider':'Enrichr / Maayan laboratory','terms':len(result),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'request_type':'Download complete public resource; no expression data or gene list uploaded'},indent=2))
    return result,path


def unique_physical_contacts(rows,members,weights,xy,sigma=20.):
    """One undirected physical pair; mean original weight over its occurrences."""
    first,last=np.triu_indices(members.shape[1],1);keys=[];values=[];n=len(xy)
    for start in range(0,len(rows),128):
        ids=np.asarray(rows[start:start+128]);m=np.asarray(members[ids]);w=physical_weights(torch.as_tensor(xy[m],dtype=torch.float32),
            torch.as_tensor(weights[ids],dtype=torch.float32),sigma,torch.as_tensor(m)).numpy()[:,first,last]
        u=np.minimum(m[:,first],m[:,last]);v=np.maximum(m[:,first],m[:,last]);keep=(u!=v)&(w>1e-8)
        keys.append((u*n+v)[keep]);values.append(w[keep])
    if not keys:return np.empty(0,int),np.empty(0,int),np.empty(0,float)
    key,inverse=np.unique(np.concatenate(keys),return_inverse=True);value=np.concatenate(values)
    mean=np.bincount(inverse,weights=value)/np.bincount(inverse)
    return key//n,key%n,mean


def contact_enrichment(u,v,weight,codes,xy,sections,ntypes,seed=52173,permutations=199,block_um=100.):
    """Within-region spatial-block label shuffle; conditions never shuffled."""
    codes=np.asarray(codes,dtype=np.int64)
    nodes=np.unique(np.r_[u,v]);inverse=np.full(len(codes),-1,int);inverse[nodes]=np.arange(len(nodes))
    uu=inverse[u];vv=inverse[v];labels=codes[nodes].copy()
    def count(c):
        a=np.minimum(c[uu],c[vv]);b=np.maximum(c[uu],c[vv])
        return np.bincount(a*ntypes+b,weights=weight,minlength=ntypes**2).reshape(ntypes,ntypes)
    obs=count(labels);blocks=np.floor(xy[nodes]/block_um).astype(int)
    keys=np.asarray([f'{s}:{a}:{b}' for s,(a,b) in zip(sections[nodes],blocks)])
    groups=[np.flatnonzero(keys==key) for key in np.unique(keys)];rng=np.random.default_rng(seed)
    null=[]
    for _ in range(permutations):
        shuffled=labels.copy()
        for ids in groups:shuffled[ids]=rng.permutation(labels[ids])
        null.append(count(shuffled))
    null=np.asarray(null);expected=null.mean(0);pseudo=max(float(weight.sum())*1e-6,1e-6)
    ratio=np.log2((obs+pseudo)/(expected+pseudo));p=(1+(null>=obs).sum(0))/(permutations+1)
    eligible=np.triu(expected>=.5);q=np.ones_like(p)
    if eligible.any():q[eligible]=multipletests(p[eligible],method='fdr_bh')[1]
    return obs,expected,ratio,p,q


def gene_effects(log_expression,genes,case_rows,background_rows):
    """Same-parent-cell-type RNA effect sizes, without cell-level disease p-values."""
    a=log_expression[case_rows].copy();b=log_expression[background_rows].copy()
    da=np.asarray((a>0).mean(0)).ravel();db=np.asarray((b>0).mean(0)).ravel()
    a.data=np.expm1(a.data);b.data=np.expm1(b.data)
    ma=np.asarray(a.mean(0)).ravel();mb=np.asarray(b.mean(0)).ravel()
    return pd.DataFrame({'gene':genes,'niche_mean_TP10K':ma,'background_mean_TP10K':mb,
        'log2FC':np.log2((ma+.1)/(mb+.1)),'niche_detection':da,'background_detection':db,
        'universe':(da>=.05)|(db>=.05)})


def panel_enrichment(effects,sets,top=100):
    universe=set(effects.loc[effects.universe,'gene']);up=effects[(effects.universe)&(effects.log2FC>=.25)&(effects.niche_detection>=.1)].nlargest(top,'log2FC')
    selected=set(up.gene);rows=[]
    for term,genes in sets.items():
        covered=genes&universe
        if len(covered)<5:continue
        overlap=selected&covered
        p=float(hypergeom.sf(len(overlap)-1,len(universe),len(covered),len(selected))) if selected else 1.
        rows.append({'pathway':term,'p_gene_overlap':p,'covered_genes':len(covered),'database_genes':len(genes),
            'panel_coverage':len(covered)/len(genes),'universe_genes':len(universe),'selected_genes':len(selected),
            'overlap':len(overlap),'overlap_genes':';'.join(sorted(overlap))})
    return pd.DataFrame(rows)


def pair_opportunity(rows,sender,receiver,codes,members,weights,xy):
    result=[]
    for start in range(0,len(rows),128):
        ids=rows[start:start+128];m=members[ids]
        w=physical_weights(torch.tensor(xy[m],dtype=torch.float32),torch.tensor(weights[ids]),20.,torch.tensor(m)).numpy()
        result.extend(np.einsum('bi,bij,bj->b',(codes[m]==sender).astype(float),w,(codes[m]==receiver).astype(float)))
    return np.asarray(result)


def ensure_downstream(source,permutations=199):
    import anndata as ad
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(8);threadpool_limits(8)
    source=Path(source);out=source/'downstream';out.mkdir(exist_ok=True);cache=input_root(source)
    sets,gmt=hallmark_sets(Path('/home/nas3/biod/xueshuailin/CCC_Phe/workspace/data/gene_sets'))
    signature={'checkpoint':hashlib.sha256((source/'gnn/checkpoint.pt').read_bytes()).hexdigest(),
        'assignments':hashlib.sha256((source/'condition_niches/assignments.csv.gz').read_bytes()).hexdigest(),
        'gene_sets':hashlib.sha256(gmt.read_bytes()).hexdigest(),'code':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'permutations':permutations,'seed':52173}
    if (out/'complete.json').exists() and json.loads((out/'complete.json').read_text())==signature:return out
    cells=pd.read_csv(source/'condition_niches/assignments.csv.gz',low_memory=False);a=ad.read_h5ad(cache/'tissue.h5ad')
    order=pd.Index(a.obs.cell_id.astype(str)).get_indexer(cells.cell_id.astype(str))
    if (order<0).any() or len(np.unique(order))!=len(a):raise ValueError('Exact physical cell alignment failed')
    a=a[order].copy();xy=cells[['x','y']].to_numpy(np.float32)
    if not np.allclose(xy,a.obsm['spatial'],atol=1e-3):raise ValueError('Coordinates disagree')
    valid=cells.analysis_center.to_numpy(bool);case=cells.condition.eq('DKD').to_numpy();labels=cells.label.to_numpy()
    types=tuple(json.loads((source/'summary.json').read_text())['cell_types']);codes=pd.Categorical(cells.cell_type,categories=types).codes
    members=np.load(cache/'members.npy');weights=np.load(cache/'weights.npy');target=np.load(cache/'ccc.npy',mmap_mode='r');features=pd.read_csv(cache/'features.csv')
    log=a.X.tocsr().astype(np.float32);totals=np.asarray(log.sum(1)).ravel();log=sp.diags(10000/np.maximum(totals,1))@log;log.data=np.log1p(log.data);log=log.tocsr()
    sp.save_npz(out/'observed_log_expression.npz',log)
    contacts=[]
    for niche in ['R1','R2','R3','R4']:
        print('Downstream contacts',niche,flush=True);rows=np.flatnonzero(valid&case&(labels==niche))
        u,v,w=unique_physical_contacts(rows,members,weights,xy);np.savez(out/f'{niche}_physical_contacts.npz',u=u,v=v,weight=w)
        obs,expected,ratio,p,q=contact_enrichment(u,v,w,codes,xy,cells.image.to_numpy(),len(types),permutations=permutations)
        for i in range(len(types)):
            for j in range(i,len(types)):
                contacts.append({'niche':niche,'type_a':types[i],'type_b':types[j],'observed_weight':obs[i,j],
                    'null_mean_weight':expected[i,j],'log2_enrichment':ratio[i,j],'p_spatial_shuffle':p[i,j],'q_spatial_shuffle':q[i,j]})
    pd.DataFrame(contacts).to_csv(out/'physical_contact_enrichment.csv',index=False)
    directed=[]
    for niche in ['R1','R2','R3','R4']:
        rows=np.flatnonzero(valid&case&(labels==niche));mean=np.asarray(target[rows]).mean(0)
        table=features[['sender','receiver']].copy();table['CCC']=mean
        group=table.groupby(['sender','receiver']).CCC.sum();total=max(float(group.sum()),1e-12)
        directed.extend({'niche':niche,'sender':s,'receiver':r,'observed_CCC':float(v),'relative_CCC':float(v/total)} for (s,r),v in group.items())
    pd.DataFrame(directed).to_csv(out/'directed_CCC_networks.csv',index=False)
    effects=[];enrich=[];states=[];groupmeans=[]
    for niche,parent in CONTEXTS:
        take=valid&case&(labels==niche)&cells.cell_type.eq(parent).to_numpy();bg=valid&case&cells.label.str.startswith('B').to_numpy()&cells.cell_type.eq(parent).to_numpy()
        if min(take.sum(),bg.sum())<30:continue
        table=gene_effects(log,np.asarray(a.var_names),np.flatnonzero(take),np.flatnonzero(bg));table['niche']=niche;table['cell_type']=parent;table['niche_cells']=take.sum();table['background_cells']=bg.sum();effects.append(table)
        ora=panel_enrichment(table,sets);ora['context']=niche+' · '+parent;enrich.append(ora)
        for group,mask in [('Healthy shared',valid&~case&cells.cell_type.eq(parent).to_numpy()),('DKD shared',bg),(niche,take)]:
            rows=np.flatnonzero(mask)
            if not len(rows):continue
            groupmeans.append({'context':niche+' · '+parent,'group':group,'cells':len(rows)})
    pd.concat(effects,ignore_index=True).to_csv(out/'same_cell_type_gene_effects.csv.gz',index=False)
    enrichment=pd.concat(enrich,ignore_index=True);enrichment['q_gene_overlap']=multipletests(enrichment.p_gene_overlap,method='fdr_bh')[1]
    enrichment.to_csv(out/'hallmark_panel_enrichment.csv',index=False);pd.DataFrame(groupmeans).to_csv(out/'expression_group_sizes.csv',index=False)
    # Gene-only state map is downstream interpretation, not niche discovery.
    rows=np.flatnonzero(valid&cells.cell_type.eq('TAL').to_numpy());x=log[rows];mean=np.asarray(x.mean(0)).ravel();var=np.asarray(x.multiply(x).mean(0)).ravel()-mean**2
    detected=np.asarray((x>0).mean(0)).ravel();eligible=np.flatnonzero((detected>=.05)&(var>.01));genes=eligible[np.argsort(-var[eligible])[:1500]]
    dense=x[:,genes].toarray();dense-=dense.mean(0);dense/=np.maximum(dense.std(0),.5)
    pcs=PCA(n_components=min(30,len(rows)-1,len(genes)-1),random_state=52173,svd_solver='randomized').fit_transform(dense)
    from umap import UMAP
    embedding=UMAP(n_neighbors=30,min_dist=.35,metric='cosine',random_state=52173,n_jobs=1,n_epochs=200).fit_transform(pcs)
    frame=cells.iloc[rows][['cell_id','condition','label','cell_type_fine','x','y']].copy();frame['source_row']=rows;frame['UMAP1']=embedding[:,0];frame['UMAP2']=embedding[:,1]
    frame['expression_group']=np.where(~case[rows],'Healthy shared',np.where(labels[rows]=='R2','DKD R2',np.where(cells.label.str.startswith('B').to_numpy()[rows],'DKD shared','DKD other R')))
    frame.to_csv(out/'TAL_expression_state_map.csv.gz',index=False);pd.Series(np.asarray(a.var_names)[genes],name='gene').to_csv(out/'TAL_state_map_genes.csv',index=False)
    # Parent-cell-conditioned injury fractions, not correlations across patients.
    for parent,injured,niche in [('TAL','iTAL','R2'),('PT','iPT','R4')]:
        ptype=cells.cell_type.eq(parent).to_numpy();fine=cells.cell_type_fine.eq(injured).to_numpy()
        for name,mask in [('Healthy whole ROI',valid&~case),('DKD whole ROI',valid&case),('DKD shared niches',valid&case&cells.label.str.startswith('B').to_numpy()),(f'DKD {niche}',valid&case&(labels==niche))]:
            ids=mask&ptype
            states.append({'parent_type':parent,'state':injured,'group':name,'parent_cells':int(ids.sum()),'injured_cells':int((ids&fine).sum()),'injured_fraction':float(fine[ids].mean()) if ids.any() else np.nan})
    pd.DataFrame(states).to_csv(out/'niche_local_and_whole_injury_states.csv',index=False)
    # R2: same sender/receiver types, paired centers, original opportunities.
    pairs=pd.read_csv(source/'condition_niches/matched_centers.csv.gz').query("niche=='R2'");qrows=pairs.case_row.to_numpy();rrows=pairs.control_row.to_numpy();lrrows=[]
    for receiver in ['Macro','CD8+']:
        si=types.index('TAL');ri=types.index(receiver);oq=pair_opportunity(qrows,si,ri,codes,members,weights,xy);orr=pair_opportunity(rrows,si,ri,codes,members,weights,xy)
        keep=(oq>1e-4)&(orr>1e-4);fids=np.flatnonzero(features.sender.eq('TAL')&features.receiver.eq(receiver));qa=np.asarray(target[qrows[keep]])[:,fids];ra=np.asarray(target[rrows[keep]])[:,fids]
        qrel=qa/np.maximum(qa.sum(1,keepdims=True),1e-12);rrel=ra/np.maximum(ra.sum(1,keepdims=True),1e-12)
        for j,fid in enumerate(fids):
            f=features.iloc[fid];lrrows.append({'sender':'TAL','receiver':receiver,'LR':f.ligand+'–'+f.receptor,'feature':int(fid),'matched_pairs':int(keep.sum()),
                'case_native_CCC':float(qa[:,j].mean()),'control_native_CCC':float(ra[:,j].mean()),'case_within_pair_fraction':float(qrel[:,j].mean()),'control_within_pair_fraction':float(rrel[:,j].mean()),
                'within_pair_difference':float(qrel[:,j].mean()-rrel[:,j].mean()),'case_detection':float((qa[:,j]>0).mean()),'control_detection':float((ra[:,j]>0).mean())})
    pd.DataFrame(lrrows).to_csv(out/'R2_same_sender_receiver_CCC.csv',index=False)
    audit={'model_refit':False,'CCC_target_changed':False,'gene_expression':'Observed integer counts → library-normalized log1p(TP10K); unique physical cells',
        'gene_effect_control':'Same parent type in DKD shared B niches; 0.1 TP10K pseudocount; no cell-level disease p-values',
        'contacts':'Original physical-pair weights, mean over repeated unordered pairs; no duplicate edge counting;100µm block-restricted type shuffles within each niche',
        'permutation_test':'Spatial arrangement conditional on this tissue and local cell-type counts; not a patient-level disease test',
        'gene_sets':str(gmt),'enrichment':'Top100 positive genes log2FC≥0.25, detection≥10%; same-type detected-panel universe; BH over all tested context/pathway pairs',
        'state_map':'PCA + UMAP of observed TAL RNA; outcome-blind variable gene selection; not a differentiation trajectory',
        'injury_states':'Published iTAL/iPT annotations, interpreted post-fit only; never encoder inputs',
        'clinical_association':'Not run: one section per condition; no invented pathology severity or survival labels'}
    (out/'audit.json').write_text(json.dumps(audit,indent=2));(out/'complete.json').write_text(json.dumps(signature,indent=2));return out
