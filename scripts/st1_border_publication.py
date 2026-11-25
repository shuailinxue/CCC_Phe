"""ST1 upper/lower C4 border comparison. Inference is exploratory within one section.

Run with the project's cccphe Python environment. CLI arguments expose paths,
block size, permutation count, bootstrap count and minimum standardized effect.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Patch
from scipy.stats import fisher_exact, chi2_contingency
from sklearn.decomposition import PCA
from statsmodels.stats.multitest import multipletests

BASE=Path('/data1/xueshuailin/CCC_Phe/workspace')
RUN=BASE/'outputs/nsclc_zhencaiwei/ST1/v2'
DATA=BASE/'data/nsclc_zhencaiwei/processed/ST1/ST1_counts.h5ad'
OUT=BASE/'artifacts/st1_upper_lower_publication'
POINTS={
'Upper':[(17400,8200),(16900,7200),(17700,6300),(18600,6000),(19700,5500),(21100,5700),(21400,6800),(21500,7600),(20700,8700),(20100,9000),(19000,9300),(17800,9200),(17400,8200),(17400,8200)],
'Lower':[(18300,10300),(18300,9700),(19200,9300),(20100,9250),(20700,9200),(20900,8350),(21600,8500),(22500,8650),(22700,10400),(22400,11900),(22200,13000),(20300,13200),(19200,12600),(18300,12300),(17700,11100),(18300,10300),(18300,10300)]}
NCOLORS=['#348f90','#d28d70','#a28ab5','#cba653','#6f8dae','#82a776']
UP='#287f80'; LO='#bd7659'

def outline(points):
    v=[(x,-y) for x,y in points]
    return MplPath(v,[MplPath.MOVETO]+[MplPath.CURVE4]*(len(v)-2)+[MplPath.CLOSEPOLY])

def bh(p):
    q=np.full(len(p),np.nan); ok=np.isfinite(p)
    if ok.any():q[ok]=multipletests(np.asarray(p)[ok],method='fdr_bh')[1]
    return q

def load_inputs(run,data,boundaries):
    a=ad.read_h5ad(data,backed='r');xy=np.asarray(a.obsm['spatial_native']);obs=a.obs[['assign.copyKAT','Tumor_boundary','nCount_RNA']].copy();a.file.close()
    coarse=np.load(run/'coarse_K4.npy');ids=np.flatnonzero(coarse==3)
    labels=np.load(run/'granularity_exploration/coarse_3_fine_K6.npy')
    if len(labels)!=len(ids) or len(xy)!=len(coarse):raise ValueError('Cell alignment mismatch')
    x,y=xy[ids,1],xy[ids,0];edge=obs.Tumor_boundary.astype(str).to_numpy()[ids]
    paths={n:outline(p) for n,p in POINTS.items()}
    mask={n:p.contains_points(np.c_[x,-y]) & np.isin(edge,boundaries) for n,p in paths.items()}
    if np.any(mask['Upper']&mask['Lower']):raise ValueError('Overlapping region masks')
    target=Path(json.loads((run/'target_cache_01.json').read_text())['folder'])
    comp=np.load(target/'composition.npy',mmap_mode='r')[ids]
    types=sorted(obs['assign.copyKAT'].astype(str).unique())
    if comp.shape[1]!=len(types):raise ValueError('Composition and cell-type count mismatch')
    return dict(xy=xy,ids=ids,x=x,y=y,edge=edge,paths=paths,mask=mask,lab=labels,comp=comp,types=types,obs=obs.iloc[ids])

def block_data(d,size,min_cells):
    record=[]
    for region in ('Upper','Lower'):
        rows=np.flatnonzero(d['mask'][region]); block=np.floor(np.c_[d['x'][rows],d['y'][rows]]/size).astype(int)
        _,inv=np.unique(block,axis=0,return_inverse=True)
        for b in range(inv.max()+1):
            local=rows[inv==b]
            if len(local)<min_cells:continue
            record.append(dict(region=region,rows=local,n=len(local),composition=d['comp'][local].mean(0),
                depth=np.log1p(pd.to_numeric(d['obs'].nCount_RNA.iloc[local]).to_numpy()).mean(),
                inner=np.mean(d['edge'][local]=='Inner 1'),
                niche=np.bincount(d['lab'][local],minlength=6)/len(local)))
    return record

def permutation_diff(v,region,nperm,rng):
    up=region==1; real=v[up].mean(axis=0)-v[~up].mean(axis=0)
    perm=np.stack([v[rng.permutation(region)==1].mean(axis=0) for _ in range(nperm)])
    # Reconstruct the complementary mean from the total for each permutation.
    diff=(perm*len(v)-v.sum(axis=0))/(~up).sum()
    p=(1+(np.abs(diff)>=np.abs(real)).sum(axis=0))/(nperm+1)
    return real,p

def composition_stats(d,blocks,nperm,nboot,rng):
    cells=[d['lab'][d['mask'][r]] for r in ('Upper','Lower')]
    counts=np.stack([np.bincount(v,minlength=6) for v in cells]); fractions=counts/counts.sum(1)[:,None]
    region=np.array([r['region']=='Upper' for r in blocks],int);v=np.stack([r['niche'] for r in blocks])
    effect,p=permutation_diff(v,region,nperm,rng);up=v[region==1];lo=v[region==0]
    n_up=int((region==1).sum()); overall_observed=np.square(effect).sum(); overall_extreme=0
    for _ in range(nperm):
        perm=rng.permutation(len(v)); difference=v[perm[:n_up]].mean(0)-v[perm[n_up:]].mean(0)
        overall_extreme+=np.square(difference).sum()>=overall_observed
    overall_p=(1+overall_extreme)/(1+nperm)
    boots=np.stack([up[rng.integers(len(up),size=len(up))].mean(0)-lo[rng.integers(len(lo),size=len(lo))].mean(0) for _ in range(nboot)])
    upper_weights=np.array([b['n'] for b in blocks if b['region']=='Upper']); lower_weights=np.array([b['n'] for b in blocks if b['region']=='Lower'])
    def weighted_boot(values,weights):
        samples=[]
        for _ in range(nboot):
            ix=rng.integers(len(values),size=len(values));samples.append(np.average(values[ix],axis=0,weights=weights[ix]))
        return np.stack(samples)
    ub=weighted_boot(up,upper_weights);lb=weighted_boot(lo,lower_weights)
    p_cell=[];orr=[]
    for j in range(6):
        table=[[counts[0,j],counts[0].sum()-counts[0,j]],[counts[1,j],counts[1].sum()-counts[1,j]]]
        odds,pv=fisher_exact(table);orr.append(odds);p_cell.append(pv)
    test=chi2_contingency(counts)
    out=pd.DataFrame({'niche':[f'C4.{i+1}' for i in range(6)],'upper_count':counts[0],'lower_count':counts[1],
        'upper_fraction':fractions[0],'lower_fraction':fractions[1],'cell_weighted_difference':fractions[0]-fractions[1],
        'upper_block_bootstrap_ci_low':np.quantile(ub,.025,axis=0),'upper_block_bootstrap_ci_high':np.quantile(ub,.975,axis=0),
        'lower_block_bootstrap_ci_low':np.quantile(lb,.025,axis=0),'lower_block_bootstrap_ci_high':np.quantile(lb,.975,axis=0),
        'odds_ratio_cell_descriptive':orr,'fisher_p_cell_descriptive':p_cell,
        'block_unweighted_difference':effect,'block_bootstrap_ci_low':np.quantile(boots,.025,axis=0),
        'block_bootstrap_ci_high':np.quantile(boots,.975,axis=0),'block_permutation_p':p})
    out['block_permutation_q_BH']=bh(out.block_permutation_p)
    return out,{'cell_chi_square':test.statistic,'cell_chi_square_p_descriptive':test.pvalue,
                'n_blocks_upper':len(up),'n_blocks_lower':len(lo),'overall_block_permutation_p_exploratory':overall_p}

def communication_stats(d,blocks,run,nperm,min_effect,rng):
    values=np.load(run/'exposure.npy',mmap_mode='r');mask=np.load(run/'exposure_mask.npy',mmap_mode='r');meta=pd.read_csv(run/'exposure_support.csv')
    if values.shape!=mask.shape or values.shape[1]!=len(meta):raise ValueError('Feature metadata mismatch')
    nfeat=len(meta);nt=len(blocks);mean=np.full((nt,nfeat),np.nan);support=np.zeros((nt,nfeat));n_supported=np.zeros((nt,nfeat),int)
    raw_sum=np.zeros((2,nfeat));raw_n=np.zeros((2,nfeat),int)
    for i,b in enumerate(blocks):
        rows=d['ids'][b['rows']];z=np.asarray(values[rows]);m=np.asarray(mask[rows],bool)&np.isfinite(z)
        n=m.sum(0);s=np.where(m,z,0).sum(0);n_supported[i]=n;support[i]=n/len(rows)
        mean[i]=np.where(n>=5,s/np.maximum(n,1),np.nan)
        region=0 if b['region']=='Upper' else 1;raw_sum[region]+=s;raw_n[region]+=n
    raw_means=np.divide(raw_sum,raw_n,out=np.full_like(raw_sum,np.nan),where=raw_n>0)
    composition=np.stack([b['composition'] for b in blocks]); pcs=PCA(n_components=3).fit_transform(np.sqrt(np.maximum(composition,0)))
    depth=np.array([b['depth'] for b in blocks]);inner=np.array([b['inner'] for b in blocks]);region=np.array([b['region']=='Upper' for b in blocks],float)
    type_index={t:i for i,t in enumerate(d['types'])}
    result=[]
    for j,m in enumerate(meta.itertuples(index=False)):
        ok=np.isfinite(mean[:,j]); group=region[ok];ng=[int((group==1).sum()),int((group==0).sum())]
        record={'upper_mean':raw_means[0,j],'lower_mean':raw_means[1,j],'raw_difference':raw_means[0,j]-raw_means[1,j],
          'upper_supported_cells':raw_n[0,j],'lower_supported_cells':raw_n[1,j],
          'upper_supported_blocks':ng[0],'lower_supported_blocks':ng[1],
          'adjusted_effect_sd':np.nan,'adjusted_p_spatial_block_exploratory':np.nan,'status':'insufficient_supported_blocks'}
        si=type_index.get(str(m.sender));ri=type_index.get(str(m.receiver))
        if min(ng)>=8 and si is not None and ri is not None:
            # Low-dimensional composition + explicit sender/receiver abundance,
            # detection depth, border type and sender-receiver support opportunity.
            X=np.column_stack([np.ones(nt),pcs,composition[:,si],composition[:,ri],depth,inner,support[:,j]])[ok]
            y=mean[ok,j];sd=y.std(ddof=1)
            if sd>1e-8:
                y=y/sd; residual_y=y-X@np.linalg.lstsq(X,y,rcond=None)[0]
                residual_g=group-X@np.linalg.lstsq(X,group,rcond=None)[0]
                den=residual_g@residual_g
                if den>1e-6 and len(y)-np.linalg.matrix_rank(X)>=5:
                    observed=(residual_g@residual_y)/den
                    # Freedman–Lane residual permutation, block as unit.
                    extreme=0
                    for _ in range(nperm):
                        perm=residual_y[rng.permutation(len(y))]
                        extreme+=abs((residual_g@perm)/den)>=abs(observed)
                    record.update(adjusted_effect_sd=observed,adjusted_p_spatial_block_exploratory=(extreme+1)/(nperm+1),status='tested')
                else:record['status']='region_collinear_or_insufficient_df'
            else:record['status']='constant_feature'
        result.append(record)
    out=pd.concat([meta.reset_index(drop=True),pd.DataFrame(result)],axis=1)
    out['adjusted_q_BH_exploratory']=bh(out.adjusted_p_spatial_block_exploratory)
    out['passes_exploratory_filter']=(out.adjusted_q_BH_exploratory<.05)&(out.adjusted_effect_sd.abs()>=min_effect)
    return out

def make_figure(d,niche,comm,out,meta):
    mpl.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7})
    fig=plt.figure(figsize=(13.5,8.8),layout='constrained');gs=fig.add_gridspec(2,2,height_ratios=[1.16,1],width_ratios=[1.1,1])
    ax=fig.add_subplot(gs[0,0]);b=fig.add_subplot(gs[0,1]);c=fig.add_subplot(gs[1,:])
    order=np.random.default_rng(2026).permutation(len(d['ids']))
    ax.scatter(d['x'][order],-d['y'][order],c=np.asarray(NCOLORS)[d['lab'][order]],s=.45,lw=0,rasterized=True)
    for name,col in [('Upper',UP),('Lower',LO)]:
        ax.add_patch(PathPatch(d['paths'][name],edgecolor=col,facecolor=col,alpha=.06,lw=0))
        ax.add_patch(PathPatch(d['paths'][name],edgecolor=col,facecolor='none',lw=1.15))
    ax.text(17450,-5800,'Upper',color=UP,fontweight='bold');ax.text(18300,-9250,'Lower',color=LO,fontweight='bold')
    ax.set(xlim=(15800,23100),ylim=(-13600,-5100));ax.set_aspect('equal');ax.axis('off');ax.set_title('a   Spatial delineation of tumor borders',loc='left',fontweight='bold',fontsize=11)
    ax.legend(handles=[Patch(color=NCOLORS[i],label=f'C4.{i+1}') for i in range(6)],loc='lower center',bbox_to_anchor=(.5,-.12),ncol=3,fontsize=7,frameon=False)
    tab=niche.sort_values('cell_weighted_difference');yy=np.arange(len(tab))
    for i,(_,r) in enumerate(tab.iterrows()):b.plot([100*r.lower_fraction,100*r.upper_fraction],[i,i],color='#c4ced0',lw=1.2,zorder=0)
    # Bootstrap limits describe retained spatial blocks; estimates include all ROI cells.
    for i,(_,r) in enumerate(tab.iterrows()):
        b.plot([100*r.lower_block_bootstrap_ci_low,100*r.lower_block_bootstrap_ci_high],[i-.10,i-.10],c=LO,lw=.8)
        b.plot([100*r.upper_block_bootstrap_ci_low,100*r.upper_block_bootstrap_ci_high],[i+.10,i+.10],c=UP,lw=.8)
    b.scatter(100*tab.lower_fraction,yy-.10,s=23,color=LO,label='Lower',zorder=2);b.scatter(100*tab.upper_fraction,yy+.10,s=23,color=UP,label='Upper',zorder=2)
    for i,(_,r) in enumerate(tab.iterrows()):b.text(1.01,i,f"q={r.block_permutation_q_BH:.3f}",transform=b.get_yaxis_transform(),va='center',fontsize=7,color='#555')
    b.set_yticks(yy,tab.niche);b.set_xlabel('Fraction of border cells (%)');b.legend(loc='upper right',frameon=False,fontsize=7);b.set_title('b   Niche distribution',loc='left',fontweight='bold',fontsize=11)
    selected=comm[comm.passes_exploratory_filter].copy()
    if selected.empty:
        tested=comm[comm.status=='tested'];qmin=tested.adjusted_q_BH_exploratory.min()
        xx=tested.adjusted_effect_sd.to_numpy(); yy=-np.log10(np.maximum(tested.adjusted_q_BH_exploratory.to_numpy(),1e-12))
        c.scatter(xx,yy,s=10,c='#8d9d9e',alpha=.45,lw=0,rasterized=True)
        c.axhline(-np.log10(.05),color='#a5aeb0',lw=.8,ls='--')
        c.axvline(-meta['effect_threshold'],color='#ccd3d3',lw=.7,ls=':');c.axvline(meta['effect_threshold'],color='#ccd3d3',lw=.7,ls=':')
        c.text(.02,.96,f'No pathways pass both thresholds  ·  {len(tested):,} tested  ·  minimum q = {qmin:.3f}',ha='left',va='top',transform=c.transAxes,fontsize=8,color='#435657')
        c.set_xlabel('Adjusted standardized effect: lower ← 0 → upper');c.set_ylabel('−log10 exploratory q');c.set_ylim(bottom=0,top=max(1.6,float(yy.max())+.2))
    else:
        selected=pd.concat([selected[selected.adjusted_effect_sd<0].nsmallest(6,'adjusted_effect_sd'),selected[selected.adjusted_effect_sd>0].nlargest(6,'adjusted_effect_sd')]).sort_values('adjusted_effect_sd')
        yy=np.arange(len(selected));names=selected.pathway.str.replace('Signaling by ','',regex=False)+' · '+selected.sender+' → '+selected.receiver
        colors=np.where(selected.adjusted_effect_sd>0,UP,LO);c.barh(yy,selected.adjusted_effect_sd,color=colors,height=.7);c.set_yticks(yy,names,fontsize=7);c.axvline(0,color='#77888c',lw=.8)
        for i,(_,r) in enumerate(selected.iterrows()):c.annotate(f"q={r.adjusted_q_BH_exploratory:.3f}",(r.adjusted_effect_sd,i),xytext=(4 if r.adjusted_effect_sd>=0 else -4,0),textcoords='offset points',ha='left' if r.adjusted_effect_sd>=0 else 'right',va='center',fontsize=6)
        c.set_xlabel('Adjusted standardized effect: lower ← 0 → upper')
    c.set_title('c   Directed communication after covariate adjustment',loc='left',fontweight='bold',fontsize=11)
    fig.savefig(out/'main_figure.pdf');fig.savefig(out/'main_figure.svg');fig.savefig(out/'main_figure.png',dpi=600);plt.close(fig)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=RUN);p.add_argument('--data',type=Path,default=DATA);p.add_argument('--out',type=Path,default=OUT);p.add_argument('--block-size',type=int,default=750);p.add_argument('--min-block-cells',type=int,default=20);p.add_argument('--permutations',type=int,default=999);p.add_argument('--bootstrap',type=int,default=2000);p.add_argument('--effect-threshold',type=float,default=.3);p.add_argument('--seed',type=int,default=40700);args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    d=load_inputs(args.run,args.data,['Inner 1','Outer 1']);blocks=block_data(d,args.block_size,args.min_block_cells);rng=np.random.default_rng(args.seed)
    print('Cells', {k:int(v.sum()) for k,v in d['mask'].items()},'blocks',len(blocks),flush=True)
    niche,overall=composition_stats(d,blocks,args.permutations,args.bootstrap,rng);niche.to_csv(args.out/'niche_composition_stats.csv',index=False)
    # Sensitivity to border definition and block scale; saved separately.
    sensitivity=[]
    for boundary in [('Inner 1','Outer 1'),('Inner 1','Inner 2','Outer 1','Outer 2')]:
        dd=load_inputs(args.run,args.data,boundary)
        for size in (500,750,1000):
            bb=block_data(dd,size,args.min_block_cells)
            if min(sum(b['region']==r for b in bb) for r in ('Upper','Lower'))<3:continue
            t,ov=composition_stats(dd,bb,args.permutations,args.bootstrap,rng);t['boundary']=' / '.join(boundary);t['block_size_native']=size;sensitivity.append(t)
    pd.concat(sensitivity,ignore_index=True).to_csv(args.out/'niche_sensitivity.csv',index=False)
    print('Fitting adjusted directed communication by spatial block',flush=True)
    comm=communication_stats(d,blocks,args.run,args.permutations,args.effect_threshold,rng);comm.to_csv(args.out/'communication_stats.csv',index=False)
    overview={**overall,'cells_upper':int(d['mask']['Upper'].sum()),'cells_lower':int(d['mask']['Lower'].sum()),'block_size_native':args.block_size,'min_block_cells':args.min_block_cells,'permutations':args.permutations,'bootstraps':args.bootstrap,'effect_threshold':args.effect_threshold,'communication_tested':int((comm.status=='tested').sum()),'communication_filtered':int(comm.passes_exploratory_filter.sum()),'assumptions':'One section; blocks are spatially dependent, regions are not replicated; all inference exploratory; coordinates and block sizes in native units.'}
    (args.out/'analysis_metadata.json').write_text(json.dumps(overview,indent=2)+'\n')
    make_figure(d,niche,comm,args.out,overview)
    legend=(f"Upper and lower tumor-border heterogeneity in one ST1 section. (a) C4 spatial labels and analysis contours chosen after inspection; border cells were restricted to author Inner 1 and Outer 1 annotations. (b) Points show cell-weighted niche fractions from all selected cells; bars are 95% block-bootstrap intervals based on retained blocks (≥{args.min_block_cells} cells), with block-size weighting. Two-sided spatial-block label permutation ({args.permutations} permutations) and Benjamini–Hochberg across six niches produce exploratory q values; block-unweighted differences and their bootstrap intervals are in the source table. (c) Directed pathway scores were compared at the spatial-block level after adjusting for three composition PCs, sender and receiver fractions, sequencing depth, author inner/outer status and feature support; two-sided residual permutation and Benjamini–Hochberg across tested pathways, with absolute standardized effect ≥{args.effect_threshold}. These within-section tests are exploratory: spatial blocks may be correlated and regions were selected after visual inspection; the results do not establish patient-level replication. Coordinates and block sizes are native units.\n")
    (args.out/'figure_legend.txt').write_text(legend)
    summary=comm[comm.passes_exploratory_filter].sort_values('adjusted_q_BH_exploratory')
    text=(f"In one section, upper and lower tumor-border regions contained {overview['cells_upper']} and {overview['cells_lower']} C4 border cells. C4.4–C4.6 represented {niche.upper_fraction.iloc[3:].sum():.1%} and {niche.lower_fraction.iloc[3:].sum():.1%} of these cells, respectively; an exploratory permutation of {overall['n_blocks_upper']} versus {overall['n_blocks_lower']} spatial blocks gave an overall p={overall['overall_block_permutation_p_exploratory']:.3g}. The regions were selected after visual inspection and represent within-section spatial heterogeneity.\n\n"
          f"Initial unadjusted descriptions suggested differences in apolipoprotein- and growth arrest/complement/midkine-associated communication. After block-level adjustment for composition, sender and receiver abundance, depth, border annotation and support, {len(summary)} of {overview['communication_tested']} tested directed features passed the exploratory q<0.05 and |effect|≥{args.effect_threshold} filter. These data do not establish an abundance-independent communication difference, causality or patient-level replication.\n")
    (args.out/'interpretation.txt').write_text(text);print(json.dumps(overview,indent=2),flush=True);print(text,flush=True)

if __name__=='__main__':main()
