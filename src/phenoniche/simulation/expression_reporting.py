"""Measured frozen-CCC and expression reconstruction figures; no ideal labels."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ..gnn.reporting import style,COLORS
from .generation import TYPES
from .expression_decoder import FrozenProgramExpressionDecoder

class ExpressionReport:
    def __init__(self,output):
        import anndata as ad
        self.output=Path(output);self.summary=json.loads((self.output/'summary.json').read_text())
        self.source=Path(self.summary['source_CCC_results']);self.adata=ad.read_h5ad(self.output/'response_injected.h5ad')
        self.xy=np.asarray(self.adata.obsm['spatial']);self.design=pd.read_csv(self.output/'response_gene_design.csv',keep_default_na=False)
        self.metrics=pd.read_csv(self.output/'expression_gene_metrics.csv');self.target=np.load(self.output/'expression_target.npy',mmap_mode='r')
        self.prediction=np.load(self.output/'expression/prediction.npy',mmap_mode='r');self.baseline=np.load(self.output/'expression/baseline_prediction.npy',mmap_mode='r')
        self.present=np.load(self.output/'expression_present.npy');self.test=np.load(self.output/'expression_split_rows.npz')['test']
        self.figures=self.output/'figures';self.figures.mkdir(exist_ok=True);style()
        self.dpi=json.loads((self.output/'config.json').read_text())['expression_config']['figure_dpi']
        # One tissue for spatial maps if mini-test uses multiple replicates.
        self.show=self.adata.obs.sample_id.to_numpy()==self.adata.obs.sample_id.iloc[0]
    def save(self,fig,name):
        fig.savefig(self.figures/(name+'.png'),dpi=self.dpi,bbox_inches='tight');fig.savefig(self.figures/(name+'.pdf'),bbox_inches='tight')
        return fig
    def spatial(self,ax,value,title,categorical=False):
        value=np.asarray(value);good=self.show&np.isfinite(value);xy=self.xy[good]
        if categorical:ax.scatter(*xy.T,c=np.asarray(COLORS)[value[good].astype(int)],s=2,lw=0,rasterized=True)
        else:
            scatter=ax.scatter(*xy.T,c=value[good],cmap='viridis',s=2,lw=0,rasterized=True);plt.colorbar(scatter,ax=ax,shrink=.6)
        ax.set(title=title,xlabel='x (µm)',ylabel='y (µm)');ax.set_aspect('equal')
    def architecture(self):
        labels=pd.read_csv(self.source/'raw_clusters.csv')['CCC_Phe GNN (fixed K)'].to_numpy()
        figure,axes=plt.subplots(1,3,figsize=(11,3.5),layout='constrained')
        self.spatial(axes[0],pd.Categorical(self.adata.obs.cell_type,categories=TYPES).codes,'Cell types',True)
        self.spatial(axes[1],labels,'Original CCC niches — frozen',True)
        exposure=np.load(self.output/'ground_truth/receiver_exposure.npy')
        self.spatial(axes[2],exposure[:,0],'MI-1 incoming LR exposure')
        return self.save(figure,'frozen_CCC_and_response_design')
    def curves(self):
        ccc=pd.read_csv(self.source/'gnn/training.csv');rna=pd.read_csv(self.output/'expression/training.csv')
        fig,axes=plt.subplots(1,2,figsize=(9,3),layout='constrained')
        for name,color in [('train',COLORS[0]),('validation',COLORS[1])]:
            axes[0].plot(ccc.epoch,ccc[name+'_ccc'],label=name,color=color)
            axes[1].plot(rna.epoch,rna[name],label=name,color=color)
        axes[0].set(title='Stage 1: original CCC training',xlabel='CCC epoch',ylabel='CCC loss')
        axes[1].set(title='Stage 2: frozen-Z expression decoder',xlabel='Expression epoch',ylabel='Scaled expression Huber')
        for ax in axes:ax.legend()
        return self.save(fig,'two_stage_loss_curves')
    def recovery(self):
        fig,axes=plt.subplots(1,4,figsize=(13,3.1),layout='constrained')
        for j,ax in enumerate(axes):
            subset=self.design[(self.design.MI==j+1)&(self.design.direction==1)]
            if subset.empty:ax.set_visible(False);continue
            t=TYPES.index(subset.receiver.iloc[0]);genes=self.adata.var_names.get_indexer(subset.gene);rows=self.test[self.present[self.test,t]]
            x=np.asarray(self.target[rows,t])[:,genes].mean(1);y=np.asarray(self.prediction[rows,t])[:,genes].mean(1)
            ax.scatter(x,y,s=5,alpha=.5,color=COLORS[j],rasterized=True)
            low=min(float(x.min()),float(y.min()));high=max(float(x.max()),float(y.max()));ax.plot([low,high],[low,high],c='#777777',lw=.7)
            ax.set(title=f'MI-{j+1} response · {subset.receiver.iloc[0]}',xlabel='Observed weighted expression',ylabel='Predicted weighted expression')
        return self.save(fig,'heldout_response_reconstruction')
    def controls(self):
        d=self.metrics;response=d[(d.kind=='synthetic_response')&(d.cell_type==d.designed_receiver)];null=d[d.kind=='negative_control']
        fig,axes=plt.subplots(1,2,figsize=(9,3.2),layout='constrained')
        groups=response.groupby('MI')[['baseline_RMSE','RMSE']].mean()
        x=np.arange(len(groups));axes[0].bar(x-.18,groups.baseline_RMSE,.35,label='Type + library baseline',color='#a7b3ba');axes[0].bar(x+.18,groups.RMSE,.35,label='Baseline + frozen Z',color=COLORS[0]);axes[0].set(xticks=x,xticklabels=[f'MI-{i}' for i in groups.index],ylabel='Held-out mean gene RMSE',title='Receiver responses');axes[0].legend(fontsize=6)
        values=[response.gain_over_type_library_baseline.dropna(),null.gain_over_type_library_baseline.dropna()]
        axes[1].boxplot(values,tick_labels=['Planted response','Negative control'],showfliers=False);axes[1].axhline(0,c='#777777',lw=.7);axes[1].set(ylabel='Gain over type + library baseline',title='Independent-response versus null control')
        return self.save(fig,'expression_baseline_and_null_controls')
    def dictionary(self):
        d=np.load(self.output/'expression/dictionary.npy');fig,axes=plt.subplots(1,len(TYPES),figsize=(16,5),layout='constrained')
        names=np.asarray(self.adata.var_names)
        limit=max(float(np.quantile(np.abs(d),.99)),1e-6)
        for t,ax in enumerate(axes):
            selected=np.argsort(-np.abs(d[t]).max(0))[:10];im=ax.imshow(d[t][:,selected],aspect='auto',cmap='RdBu_r',vmin=-limit,vmax=limit)
            ax.set(title=TYPES[t],ylabel='CCC program',xlabel='Highest-loading genes');ax.set_xticks(np.arange(len(selected)),names[selected],rotation=90,fontsize=5)
        fig.colorbar(im,ax=axes,label='Signed log-count loading',shrink=.6)
        return self.save(fig,'type_conditioned_gene_dictionary')
    def response_maps(self):
        subset=self.design[(self.design.MI==1)&(self.design.direction==1)];t=TYPES.index(subset.receiver.iloc[0]);genes=self.adata.var_names.get_indexer(subset.gene)
        observed=np.asarray(self.target[:,t])[:,genes].mean(1);predicted=np.asarray(self.prediction[:,t])[:,genes].mean(1)
        observed[~self.present[:,t]]=np.nan;predicted[~self.present[:,t]]=np.nan
        fig,axes=plt.subplots(1,2,figsize=(8,4),layout='constrained')
        self.spatial(axes[0],observed,'MI-1 receiver response: observed');self.spatial(axes[1],predicted,'MI-1 receiver response: reconstructed')
        return self.save(fig,'weighted_response_spatial_maps')
    def program_ablation(self):
        """Decoder sensitivity only, not a causal gene-knockout prediction."""
        import torch
        pre=np.load(self.output/'expression/preprocessing.npz');model=FrozenProgramExpressionDecoder(*(pre[k] for k in ('program_mean','program_scale','baseline','feature_scale','feature_mask')))
        saved=torch.load(self.output/'expression/checkpoint.pt',map_location='cpu',weights_only=False);model.load_state_dict(saved['model_state']);model.eval()
        match=pd.read_csv(self.output/'synthetic_module_recovery.csv');program=int(match.loc[match.MI==1,'program'].iloc[0])-1
        z=np.asarray(np.load(self.source/'gnn/Z.npy',mmap_mode='r')[self.test]).copy();z[:,program]=0
        norm=np.load(self.output/'expression_normalization.npy')[self.test]
        with torch.no_grad():modified=model(torch.from_numpy(z),torch.from_numpy(norm)).numpy()
        change=modified-np.asarray(self.prediction[self.test]);records=[]
        for t,kind in enumerate(TYPES):
            valid=self.present[self.test,t]
            for g,name in enumerate(self.adata.var_names):
                records.append({'program':program+1,'cell_type':kind,'gene':name,'mean_predicted_change':float(change[valid,t,g].mean()) if valid.any() else np.nan})
        frame=pd.DataFrame(records);frame.to_csv(self.output/'program_ablation_sensitivity.csv',index=False)
        return frame
