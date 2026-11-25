"""Describe the saved K12 tumor niches; no fitting, relabeling or notebook edits."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import anndata as ad

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'src'))
from phenoniche.cohorts.crc_neutrophil_atlas import ROOT,OUTPUT

BASE=OUTPUT/'joint_five'
OUT=BASE/'tumor_niche_analysis'
GROUPS=(1,10,12)


def run():
    import torch
    from torch.nn import functional as F
    torch.set_num_threads(6)
    OUT.mkdir(parents=True,exist_ok=True)
    figure=json.loads((BASE/'figures/umap_K12.json').read_text())
    clusters=Path(figure['clustering']);fit=clusters.parents[1]
    matrices=Path(json.loads((fit/'complete.json').read_text())['matrices'])
    a=ad.read_h5ad(ROOT/'processed/CRC_selected_five.h5ad')
    labels=np.load(clusters/'labels.npy')+1;ty=a.obs.cell_type.astype(str).to_numpy()
    samples=list(pd.unique(a.obs['sample']));sample=a.obs['sample'].astype(str).to_numpy()
    cats=json.loads((matrices/'complete.json').read_text())['categories']
    comp=np.load(matrices/'composition.npy',mmap_mode='r')
    counts=pd.crosstab([sample,ty],labels);counts.to_csv(OUT/'sample_type_niche_counts.csv')
    rows=[]
    for k in range(1,13):
        ids=np.flatnonzero(labels==k);cancer=ty[ids]=='Cancer cell';v=np.asarray(comp[ids]).mean(0)
        row=dict(niche=f'N{k}',cells=len(ids),cancer_cells=int(cancer.sum()),central_cancer_fraction=float(cancer.mean()))
        for t in cats:row['local_'+t]=float(v[cats.index(t)])
        for s in samples:row[s]=int(np.sum(sample[ids]==s))
        if cancer.any():
            v=np.asarray(comp[ids[cancer]]).mean(0)
            for t in cats:row['cancer_center_local_'+t]=float(v[cats.index(t)])
        rows.append(row)
    summary=pd.DataFrame(rows);summary.to_csv(OUT/'niche_overview.csv',index=False)
    # Summaries stratified by specimen avoid confusing sample differences with niche effects.
    rows=[]
    for s in samples:
        denom=np.sum((sample==s)&(ty=='Cancer cell'))
        for k in GROUPS:
            mask=(sample==s)&(labels==k);cc=mask&(ty=='Cancer cell')
            v=np.asarray(comp[mask]).mean(0)
            row=dict(specimen=s,niche=f'N{k}',cells=int(mask.sum()),cancer_cells=int(cc.sum()),
                     fraction_of_specimen_cancer=float(cc.sum()/max(denom,1)))
            for t in cats:row['local_'+t]=float(v[cats.index(t)])
            rows.append(row)
    pd.DataFrame(rows).to_csv(OUT/'tumor_niches_by_specimen.csv',index=False)

    features=pd.read_csv(matrices/'features.csv');effective=np.load(fit/'feature_mask.npy')
    # Named axes are a descriptive, declared subset, not a new training feature selection.
    axes={('FN1','CD44'),('FN1','SDC1'),('CXCL12','CXCR4'),('CXCL16','CXCR6'),
          ('CCL5','CCR1'),('CCL5','CCR4'),('CCL5','CCR5'),('CD86','CD28'),('CD86','CTLA4'),
          ('AREG','EGFR'),('EGF','EGFR'),('VEGFA','FLT1'),('CD40LG','CD40'),
          ('IL1A','IL1R1'),('IL1B','IL1R1'),('CD274','PDCD1'),('PDCD1LG2','PDCD1'),
          ('CSF1','CSF1R'),('ICAM1','ITGAL')}
    use=np.array([effective[i] and ((r.ligand,r.receptor) in axes) for i,r in features.iterrows()])
    cols=np.flatnonzero(use);features=features.loc[use].reset_index(names='feature_index')
    f=len(features);t=len(cats)
    pair=np.array([cats.index(r.sender)*t+cats.index(r.receiver) for r in features.itertuples()])
    observed=np.load(matrices/'ccc.npy',mmap_mode='r');opp=np.load(matrices/'opportunity.npy',mmap_mode='r')
    h=np.load(fit/'gnn/shared.npy',mmap_mode='r')
    saved=torch.load(fit/'gnn/checkpoint.pt',map_location='cpu',weights_only=False)['model_state']
    # The hidden layer is unchanged; output coordinates are independent scalar predictions.
    w0,b0=saved['ccc_head.0.weight'],saved['ccc_head.0.bias']
    w1,b1=saved['ccc_head.2.weight'][cols],saved['ccc_head.2.bias'][cols]
    scale=saved['feature_scale'][cols]*saved['feature_mask'][cols]
    ng=len(samples)*len(GROUPS)*2
    n=np.zeros((ng,f),np.int64);obs_sum=np.zeros((ng,f));pred_sum=np.zeros((ng,f));positive=np.zeros((ng,f))
    raw_sum=np.zeros((ng,f));raw_pred=np.zeros((ng,f));raw_count=np.zeros(ng,np.int64)
    print('Aggregate observed/frozen predicted named CCC axes:',f,'features',flush=True)
    for lo in range(0,len(a),8192):
        hi=min(lo+8192,len(a));z=np.asarray(observed[lo:hi][:,cols]);op=np.asarray(opp[lo:hi][:,pair])>1
        with torch.no_grad():
            v=torch.as_tensor(np.array(h[lo:hi],copy=True))
            pred=(F.softplus(F.linear(F.gelu(F.linear(v,w0,b0)),w1,b1))*scale).numpy()
        for si,s in enumerate(samples):
            for ki,k in enumerate(GROUPS):
                base=(sample[lo:hi]==s)&(labels[lo:hi]==k)
                for scope in [0,1]:
                    take=base if scope==0 else base&(ty[lo:hi]=='Cancer cell')
                    if not take.any():continue
                    g=(si*3+ki)*2+scope;valid=op[take]
                    n[g]+=valid.sum(0);obs_sum[g]+=(z[take]*valid).sum(0,dtype=np.float64)
                    pred_sum[g]+=(pred[take]*valid).sum(0,dtype=np.float64)
                    positive[g]+=((z[take]>0)&valid).sum(0)
                    raw_sum[g]+=z[take].sum(0,dtype=np.float64);raw_pred[g]+=pred[take].sum(0,dtype=np.float64)
                    raw_count[g]+=take.sum()
    records=[]
    for si,s in enumerate(samples):
        for ki,k in enumerate(GROUPS):
            for scope in [0,1]:
                g=(si*3+ki)*2+scope
                for j,row in features.iterrows():
                    r=row.to_dict();r.update(specimen=s,niche=f'N{k}',scope='all centers' if scope==0 else 'cancer centers',
                        cells=int(raw_count[g]),opportunity_cells=int(n[g,j]),observed_sum=float(obs_sum[g,j]),
                        predicted_sum=float(pred_sum[g,j]),positive_cells=int(positive[g,j]),
                        observed_conditional_mean=float(obs_sum[g,j]/max(n[g,j],1)),
                        predicted_conditional_mean=float(pred_sum[g,j]/max(n[g,j],1)),
                        observed_positive_fraction=float(positive[g,j]/max(n[g,j],1)),
                        observed_all_mean=float(raw_sum[g,j]/max(raw_count[g],1)),
                        predicted_all_mean=float(raw_pred[g,j]/max(raw_count[g],1)))
                    records.append(r)
    table=pd.DataFrame(records);table.to_csv(OUT/'named_CCC_by_specimen_and_niche.csv',index=False)
    keys=['scope','niche','feature_index','sender','receiver','ligand','receptor']
    pooled=table.groupby(keys,as_index=False)[['cells','opportunity_cells','observed_sum','predicted_sum','positive_cells']].sum()
    for dest,source in [('observed_mean','observed_sum'),('predicted_mean','predicted_sum'),('positive_fraction','positive_cells')]:
        pooled[dest]=pooled[source]/pooled.opportunity_cells.clip(lower=1)
    pooled.to_csv(OUT/'named_CCC_niche_pooled.csv',index=False)
    print(summary.loc[summary.niche.isin(['N1','N10','N12']),['niche','cells','central_cancer_fraction','local_Cancer cell','local_T cell','local_Macrophage','local_Fibroblast']].to_string(index=False),flush=True)
    jsave=dict(clustering=str(clusters),fit=str(fit),matrices=str(matrices),K=12,groups=list(GROUPS),
               model_frozen=True,retraining=False,notebook_modified=False,opportunity_threshold=1.,
               statistics='Descriptive; overlapping neighborhoods are not independent biological replicates',
               CCC='Native original counts/unchanged directed target; frozen direct MLP prediction in native units',
               axes='Declared known LR subset; unavailable/masked axes omitted, never manufactured')
    (OUT/'analysis_protocol.json').write_text(json.dumps(jsave,indent=2))
    plot_summary(summary,table)


def plot_summary(summary,table):
    """Descriptive evidence: composition transition and within-specimen CCC contrasts."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from phenoniche.cohorts.crc_neutrophil_atlas import COLORS
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,3,figsize=(15.5,4.9));fig.subplots_adjust(left=.05,right=.99,bottom=.25,top=.79,wspace=.34)
    names=['N1','N10','N12'];tab=summary.set_index('niche').loc[names]
    left=np.zeros(3)
    types=['Cancer cell','T cell','Macrophage','Fibroblast']
    for ty in types+['Other cells']:
        v=tab['local_'+ty].to_numpy()*100 if ty!='Other cells' else 100-sum(tab['local_'+t].to_numpy()*100 for t in types)
        axs[0].barh(np.arange(3),v,left=left,color=COLORS.get(ty,'#D5D5D5'),label=ty,height=.64)
        for i,value in enumerate(v):
            if value>=8:axs[0].text(left[i]+value/2,i,f'{value:.1f}%',ha='center',va='center',fontsize=9)
        left+=v
    axs[0].set_yticks(range(3),names);axs[0].invert_yaxis();axs[0].set_xlim(0,100)
    axs[0].set_xlabel('Mean local pair-weighted cell fraction (%)');axs[0].set_title('Local environment composition')
    axs[0].legend(loc='upper center',bbox_to_anchor=(.5,-.2),ncol=2,frameon=False,fontsize=9)
    data=table[(table.scope=='all centers')&table.sender.eq('Fibroblast')&table.receiver.eq('Cancer cell')&
               table.ligand.eq('FN1')&table.receptor.eq('CD44')&~table.specimen.eq('crc_j_n')]
    samples=['crc_a_c','crc_l_c','crc_a_m','crc_b_m'];colors={s:plt.get_cmap('tab10')(i) for i,s in enumerate(samples)}
    foldrows=[]
    for ax,key,title in [(axs[1],'observed_conditional_mean','Observed directed CCC'),
                         (axs[2],'predicted_conditional_mean','Frozen-model prediction')]:
        z=data.pivot(index='specimen',columns='niche',values=key)
        for s in samples:
            vals=z.loc[s,names].to_numpy()/z.loc[s,'N1']
            ax.plot(range(3),vals,marker='o',color=colors[s],label=s,lw=1.6,ms=5)
            for n,v in zip(names,vals):foldrows.append(dict(specimen=s,niche=n,source=key,fold_over_N1=float(v)))
        ax.set_xticks(range(3),names);ax.set_ylim(.8,2.3);ax.set_ylabel('Mean CCC / same-specimen N1')
        ax.axhline(1,c='#999',ls=':',lw=1);ax.set_title(title)
    axs[2].legend(loc='upper center',bbox_to_anchor=(.5,-.2),ncol=2,frameon=False,fontsize=9)
    fig.suptitle('Tumor environments and FN1–CD44 · fibroblast → cancer',fontsize=14,y=.97)
    fig.text(.53,.035,'CCC means include only environments with the corresponding physical-pair opportunity >1.\nFour tumor specimens from three patients; descriptive effects, no independent-replicate significance claim.',
             ha='center',fontsize=8.5,color='#555')
    for ext in ['png','pdf']:fig.savefig(OUT/f'tumor_niche_evidence.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)
    pd.DataFrame(foldrows).to_csv(OUT/'FN1_CD44_within_specimen_fold_changes.csv',index=False)


if __name__=='__main__':run()
