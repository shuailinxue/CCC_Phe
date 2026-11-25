"""Read-only audit of existing PhenoMap TNBC output; no training or relabeling.

All audit products go to CCC_Phe NAS workspace. Spot reconstruction is strictly
in-sample consistency, never single-cell ground truth. Stream the 2GB CSV once.
"""
from pathlib import Path
import json,pickle,re
import numpy as np,pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SOURCE=Path('/home/nas3/biod/xueshuailin/New_Projects/data/HBC/TNBC_ST/Processed_Samples/TNBC1_CN1_C2')
OUT=Path('/home/nas3/biod/xueshuailin/CCC_Phe/workspace/outputs/phenomap_tnbc_audit/TNBC1_CN1_C2')
MARKERS={'T_cells':['CD3D','CD3E','TRAC','CD247','CD8A','IL7R','CCL5','NKG7'],
 'B_cells':['MS4A1','CD79A','CD79B','CD37','CD74','HLA-DRA','MZB1'],
 'Myeloid':['LYZ','TYROBP','FCER1G','C1QA','C1QB','C1QC','CD68','CD14'],
 'Endothelial':['PECAM1','VWF','KDR','EMCN','RAMP2'],
 'Stromal':['COL1A1','COL1A2','COL3A1','DCN','LUM','ACTA2','CXCL12'],
 'Epithelial':['EPCAM','KRT8','KRT18','KRT19','KRT5','KRT14'],
 'LR_examples':['CXCL9','CXCL10','CXCL11','CXCL12','CCL2','CCL5','TGFB1','IL6','IFNG','SPP1','MIF','APP','CXCR4','CD74','PDCD1','CD274']}


def column_correlation(a,b):
    a=a-a.mean(0);b=b-b.mean(0);den=np.sqrt((a*a).sum(0)*(b*b).sum(0))
    return np.divide((a*b).sum(0),den,out=np.full(a.shape[1],np.nan),where=den>1e-12)


def quantiles(x):
    x=np.asarray(x);x=x[np.isfinite(x)]
    return dict(zip(['min','q10','q25','median','q75','q90','max'],map(float,np.quantile(x,[0,.1,.25,.5,.75,.9,1])))) if len(x) else {}


def main():
    threadpool_limits(8);OUT.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(52173)
    path=SOURCE/'output/pre_ours_pipeline.csv';header=pd.read_csv(path,nrows=0);genes=header.columns[1:].to_numpy()
    meta=pd.read_hdf(SOURCE/'histology_cell_multifeatures.h5','metadata');ids=meta.index.to_numpy();n=len(meta);f=len(genes)
    loc=pd.read_csv(SOURCE/'locs.csv',index_col=0);xy=meta[['center_x','center_y']].to_numpy();diam=float((SOURCE/'spot_diameter_pixel.txt').read_text())
    distance,spot=cKDTree(loc[['x','y']]).query(xy);inside=distance<=diam/2
    spot_spacing=cKDTree(loc[['x','y']]).query(loc[['x','y']],k=2)[0][:,1]
    if spot_spacing.min()<diam:raise ValueError('Overlapping spots: nearest-only aggregation invalid')
    counts=np.bincount(spot[inside],minlength=len(loc));target=pd.read_csv(SOURCE/'cnts.csv',index_col=0,usecols=['spots']+list(genes)).reindex(loc.index)
    if target.isna().any().any():raise ValueError('Missing raw spot/gene alignment')
    target=target[genes].to_numpy(np.float64)
    with (SOURCE/'output/scale.pkl').open('rb') as h:scaler=pickle.load(h)
    if len(scaler.data_min_)!=f:raise ValueError('Scaler feature count disagrees')
    ct=pd.read_csv(SOURCE/'output/GT_Celltypist.csv',index_col=0).reindex(ids).iloc[:,0]
    ig=pd.read_csv(SOURCE/'output/GT_ingest.csv',index_col=0).reindex(ids).iloc[:,0]
    ens=pd.read_csv(SOURCE/'output/GT_二合一.csv',index_col=0).reindex(ids).Ensemble_Label
    if ct.isna().any() or ig.isna().any() or ens.isna().any():raise ValueError('Annotation ID mismatch')
    prob=pd.read_csv(SOURCE/'output/GT_Celltypist_probabilities.csv',index_col=0).reindex(ids)
    sample_rows=np.sort(rng.choice(n,min(8192,n),replace=False));sample=np.zeros((len(sample_rows),f),np.float32)
    marker_names=sorted(set(g for group in MARKERS.values() for g in group)&set(genes));mi=pd.Index(genes).get_indexer(marker_names)
    marker=np.zeros((n,len(mi)),np.float32);positive_total=np.zeros(n,np.float32)
    sums=np.zeros((len(loc),f),np.float64);clipped=np.zeros_like(sums);negative=np.zeros(f,np.int64);substantial=np.zeros(f,np.int64)
    total=np.zeros(f);total2=np.zeros(f);maxima=np.full(f,-np.inf);minima=np.full(f,np.inf);nnz=np.zeros(n,int);offset=0
    for chunk in pd.read_csv(path,index_col=0,chunksize=4096,dtype={g:'float32' for g in genes}):
        rows=np.arange(offset,offset+len(chunk));np.testing.assert_array_equal(chunk.index,ids[rows]);x=chunk.to_numpy()
        if not np.isfinite(x).all():raise ValueError('Non-finite predicted RNA')
        negative+=(x<0).sum(0);substantial+=(x<-.1).sum(0);total+=x.sum(0,dtype=np.float64);total2+=(x.astype(np.float64)**2).sum(0)
        maxima=np.maximum(maxima,x.max(0));minima=np.minimum(minima,x.min(0));xp=np.maximum(x,0)
        nnz[rows]=(xp>.1).sum(1);positive_total[rows]=xp.sum(1);marker[rows]=xp[:,mi]
        use=inside[rows];np.add.at(sums,spot[rows[use]],x[use]);np.add.at(clipped,spot[rows[use]],xp[use])
        sr=sample_rows[(sample_rows>=offset)&(sample_rows<offset+len(chunk))];sample[np.searchsorted(sample_rows,sr)]=xp[sr-offset]
        offset+=len(chunk)
        if offset%32768==0:print('Audited predicted cells',offset,flush=True)
    if offset!=n:raise ValueError('Cell count mismatch')
    print('Streaming complete',n,f,flush=True)
    # Training sums MINMAX cell predictions, inverse transformed only AFTER sum.
    # Saved cell CSV instead inverses each cell: sum(saved) adds N*spot_min.
    restored=sums-(counts[:,None]-1)*scaler.data_min_[None,:]
    keep=counts>0;truth=target[keep];native=sums[keep];restored=restored[keep];den=np.maximum(scaler.data_range_,1.)
    density_baseline=counts[keep,None]/counts[keep].mean()*truth.mean(0)
    rows=[];reconstruction={}
    for name,estimate in [('saved_cell_sum',native),('recover_training_units',restored),('cell_density_baseline',density_baseline)]:
        pcc=column_correlation(estimate,truth);spot_pcc=column_correlation(estimate.T,truth.T)
        mse=float(np.mean(((estimate-truth)/den)**2));null=float(np.mean(((truth-truth.mean(0))/den)**2))
        reconstruction[name]={'gene_PCC':quantiles(pcc),'spot_PCC':quantiles(spot_pcc),'scaled_MSE':mse,'gain_vs_mean_target':1-mse/max(null,1e-12)}
        rows.extend({'gene':g,'method':name,'PCC':v} for g,v in zip(genes,pcc))
    pd.DataFrame(rows).to_csv(OUT/'spot_reconstruction_by_gene.csv',index=False)
    pd.DataFrame(restored,index=loc.index[keep],columns=genes).to_csv(OUT/'reaggregated_training_units.csv.gz')
    geneqc=pd.DataFrame({'gene':genes,'negative_fraction':negative/n,'less_than_minus_point1':substantial/n,'mean':total/n,
        'std':np.sqrt(np.maximum(total2/n-(total/n)**2,0)),'min':minima,'max':maxima,'spot_min':scaler.data_min_})
    geneqc.to_csv(OUT/'predicted_gene_QC.csv',index=False)
    rawgenes=set(pd.read_csv(SOURCE/'cnts.csv',nrows=0).columns[1:]);coverage=[]
    for group,gs in MARKERS.items():
        coverage.extend({'group':group,'gene':g,'in_prediction':g in genes,'in_raw_ST':g in rawgenes} for g in gs)
    pd.DataFrame(coverage).to_csv(OUT/'marker_coverage.csv',index=False)
    atlas=pd.read_csv(Path(__file__).resolve().parents[1]/'data/commuspace_human_lr_atlas.tsv',sep='\t')
    predgenes=set(genes)
    def supported(value,universe):return all(g in universe for g in re.split(r'[&+_| ]+',str(value)))
    rawmask=np.asarray([supported(l,rawgenes) and supported(r,rawgenes) for l,r in zip(atlas.ligand,atlas.receptor)])
    predmask=np.asarray([supported(l,predgenes) and supported(r,predgenes) for l,r in zip(atlas.ligand,atlas.receptor)])
    atlas['raw_ST_covered']=rawmask;atlas['predicted_covered']=predmask;atlas.to_csv(OUT/'LR_coverage.csv',index=False)
    profile=np.log1p(sample/np.maximum(sample.sum(1,keepdims=True),1e-8)*10000)
    pca=PCA(30,random_state=52173,svd_solver='randomized').fit(profile)
    center=profile-profile.mean(1,keepdims=True);center/=np.maximum(np.linalg.norm(center,axis=1,keepdims=True),1e-8)
    _,near=cKDTree(xy[sample_rows]).query(xy[sample_rows],k=2)
    nncorr=(center*center[near[:,1]]).sum(1);rand=rng.integers(0,len(sample_rows),len(sample_rows));randomcorr=(center*center[rand]).sum(1)
    totals=sample.sum(1);mito=np.asarray([g.startswith('MT-') for g in genes]);mitofrac=sample[:,mito].sum(1)/np.maximum(totals,1e-8)
    means=[]
    # Within predicted labels, this is consistency, never accuracy validation.
    for label in sorted(ens.unique()):
        mask=ens.to_numpy()==label
        for j,g in enumerate(marker_names):means.append({'label':label,'gene':g,'cells':int(mask.sum()),'mean_TP10K':float((marker[mask,j]/np.maximum(positive_total[mask],1e-8)*10000).mean())})
    pd.DataFrame(means).to_csv(OUT/'marker_expression_by_inferred_label.csv',index=False)
    table=meta.copy();table['nearest_spot']=loc.index.to_numpy()[spot];table['distance_px']=distance;table['inside_measured_spot']=inside
    table['Celltypist']=ct;table['KNN_transfer']=ig;table['Ensemble']=ens;table['positive_total_1000genes']=positive_total;table['genes_above_point1']=nnz
    table.to_csv(OUT/'cell_QC.csv.gz')
    pd.crosstab(ct,ig).to_csv(OUT/'annotation_agreement.csv')
    markdict={k:{'measured':sorted(set(gs)&predgenes),'missing':sorted(set(gs)-predgenes)} for k,gs in MARKERS.items()}
    report={'source':str(SOURCE),'audit_only':True,'source_files_changed':False,'predicted_cells':n,'predicted_genes':f,'raw_spots':len(loc),'raw_genes':len(rawgenes),
        'finite_predictions':True,'negative_fraction':float(negative.sum()/(n*f)),'below_minus_point1_fraction':float(substantial.sum()/(n*f)),
        'nonzero_spot_min_genes':int((scaler.data_min_>0).sum()),'spot_radius_px':diam/2,'spot_spacing_px':quantiles(spot_spacing),
        'cells_inside_measured_spots':int(inside.sum()),'inside_fraction':float(inside.mean()),'cells_per_spot':quantiles(counts),
        'crop_size_px':quantiles(meta.crop_size),'crop_below_10px':float((meta.crop_size<10).mean()),'nearest_cell_distance_px':quantiles(cKDTree(xy).query(xy,k=2)[0][:,1]),
        'annotation_agreement':float(ct.eq(ig).mean()),'ensemble_distribution':ens.value_counts().to_dict(),
        'Celltypist_distribution':ct.value_counts().to_dict(),'KNN_distribution':ig.value_counts().to_dict(),
        'celltypist_probabilities':'Independent one-vs-rest outputs; NOT a softmax over types; high values in multiple classes do not by themselves imply a file error',
        'probability_row_sums':quantiles(prob.sum(1)),'top_probability_margin':quantiles(np.sort(prob.to_numpy(),axis=1)[:,-1]-np.sort(prob.to_numpy(),axis=1)[:,-2]),
        'LR_total':len(atlas),'LR_raw_ST_covered':int(rawmask.sum()),'LR_predicted_covered':int(predmask.sum()),'markers':markdict,
        'reconstruction':reconstruction,'expression_PC_variance':list(map(float,pca.explained_variance_ratio_[:10])),
        'nearest_sample_cell_profile_PCC':quantiles(nncorr),'random_sample_cell_profile_PCC':quantiles(randomcorr),'sample_mito_fraction_1000genes':quantiles(mitofrac),
        'gene_variability':'Clipped predicted RNA, total-normalized across the 1000 available genes; sampled8192 cells; not true UMI measurements',
        'reconstruction_scope':'All available original spots; IN-SAMPLE, not held-out; full-graph predictions may differ from spot-local training graphs',
        'scale_issue':'Training sums normalized cell predictions; exported CSV inverse-transforms per cell. Correct in-sample spot units are sum(saved_cells)-(N_cells-1)*spot_min. This is an algebraic audit, not a validated cell-expression correction.',
        'annotation_scope':'Both annotation methods consume inferred RNA and are not independent experimental ground truth; filenames GT do not change this.'}
    (OUT/'audit.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ['markers','reconstruction']},indent=2),flush=True)
    print(json.dumps(reconstruction,indent=2),flush=True)
    # Small diagnostic plate, no morphology image manipulation or training.
    plt.rcParams.update({'pdf.fonttype':42,'svg.fonttype':'none','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(12,9),layout='constrained')
    labels=sorted(ens.unique());palette=dict(zip(labels,['#447DA4','#C37068','#729F88','#D4AC60','#927CAD','#A5B5BC','#8A8B8D','#D9DDE1']))
    axes[0,0].scatter(xy[:,0],xy[:,1],c=[palette[t] for t in ens],s=.3,lw=0,rasterized=True);axes[0,0].set(title='Inferred cell labels · original image pixels',aspect='equal');axes[0,0].invert_yaxis()
    from matplotlib.lines import Line2D
    axes[0,0].legend(handles=[Line2D([],[],marker='o',ls='',color=palette[t],label=f'{t}: {(ens==t).sum():,}',markersize=4) for t in labels],fontsize=7,loc='upper left',bbox_to_anchor=(1,.95))
    for method,color in [('saved_cell_sum','#C37068'),('recover_training_units','#447DA4'),('cell_density_baseline','#9FAAA5')]:
        vals=pd.DataFrame(rows).query('method==@method').PCC.dropna();axes[0,1].hist(vals,bins=np.linspace(-1,1,35),histtype='step',color=color,label=method,lw=1.3)
    axes[0,1].set(title='Reaggregation consistency · NOT held-out',xlabel='Per-gene spot Pearson r',ylabel='Genes');axes[0,1].legend(fontsize=7)
    cov=pd.DataFrame(coverage);groups=list(MARKERS)
    values=[int(cov[cov.group==g].in_prediction.sum()) for g in groups];missing=[len(MARKERS[g])-v for g,v in zip(groups,values)]
    axes[1,0].barh(groups,values,color='#447DA4',label='Predicted');axes[1,0].barh(groups,missing,left=values,color='#E0E4E7',label='Not predicted');axes[1,0].set(title='Key marker and LR gene availability',xlabel='Genes');axes[1,0].legend(fontsize=8)
    genes_show=[g for g in ['EPCAM','COL1A1','LUM','CD74','MS4A1','CD79A','CD3D','CD3E','TRAC','LYZ','PECAM1'] if g in marker_names]
    wide=pd.DataFrame(means).pivot(index='label',columns='gene',values='mean_TP10K').reindex(index=labels,columns=genes_show)
    v=np.log1p(wide);v=v/np.maximum(v.max(0),1e-8);im=axes[1,1].imshow(v,cmap='BuPu',aspect='auto',vmin=0,vmax=1)
    axes[1,1].set(xticks=range(len(genes_show)),xticklabels=genes_show,yticks=range(len(labels)),yticklabels=labels,title='Observed consistency of inferred RNA and labels')
    plt.setp(axes[1,1].get_xticklabels(),rotation=45,ha='right');fig.colorbar(im,ax=axes[1,1],shrink=.7,label='Per-gene relative log mean TP10K')
    for ext in ['png','pdf']:fig.savefig(OUT/f'quality_audit.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)
    print('Audit saved',OUT,flush=True)


if __name__=='__main__':main()
