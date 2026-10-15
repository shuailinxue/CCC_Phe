"""Compact, measured figures for CCC-priority joint expression simulation.

Figure contract: test whether niches and molecular signatures survive joint
training. Matched observed/reconstructed scales, raw cluster IDs, held-out gene
metrics and CCC-only controls prevent an attractive map from claiming success.
Python quantitative grids, editable PDF/SVG and 300-dpi PNG on NAS.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from .expression_reporting import ExpressionReport
from .generation import TYPES
from ..gnn.reporting import COLORS


class JointExpressionReport(ExpressionReport):
    def __init__(self,output):
        super().__init__(output)
        plt.rcParams.update({'pdf.fonttype':42,'svg.fonttype':'none','font.size':8})
        self.z=np.load(self.output/'gnn/Z.npy');self.h=np.load(self.output/'gnn/H.npy')
        self.labels=pd.read_csv(self.output/'raw_clusters.csv')['CCC_Phe GNN (selected K)'].to_numpy()
        cc_source=Path(self.summary.get('CCC_only_niche_results_dir',self.source))
        self.old_labels=pd.read_csv(cc_source/'raw_clusters.csv')['CCC_Phe GNN (fixed K)'].to_numpy()
        self.truth=self.adata.obs.true_niche.to_numpy()
        self.model_label='Joint GNN' if self.summary['joint_model_accepted'] else 'Frozen-Z GNN'
        self.composition=pd.read_csv(self.output/'niche_cell_composition.csv',index_col=0)
        self.top_ccc=pd.read_csv(self.output/'niche_top10_CCC.csv')
        true_h=np.load(self.source/'ground_truth/MI_dictionary.npy');den=np.maximum((true_h*true_h).sum(1),1e-12)
        c=np.load(self.output/'anchor_inputs/input/ccc.npy',mmap_mode='r')
        self.true_signal=np.asarray(c@true_h.T)/den
        self.recovered_signal=self.z@(self.h@true_h.T)/den
        self.mi=np.load(self.source/'ground_truth/MI_activity.npy')
        specs=pd.read_csv(self.source/'ground_truth/MI_programs.csv');self.specs=specs
        self.limits=np.quantile(np.r_[self.true_signal,self.recovered_signal],.99,axis=0)
        self.gene_scores=pd.read_csv(self.output/'expression_gene_metrics.csv')

    def save(self,fig,name):
        for ext in ('png','pdf','svg'):fig.savefig(self.figures/(name+'.'+ext),dpi=self.dpi,bbox_inches='tight')
        return fig

    def panel(self,ax,value,title,categorical=False,vmax=None,names=None):
        value=np.asarray(value);valid=self.show&np.isfinite(value)
        if categorical:
            colors=list(COLORS)
            if len(np.unique(value[valid]))>len(colors):colors=list(plt.get_cmap('tab20').colors)
            cmap=ListedColormap(colors)
            ax.scatter(*self.xy[valid].T,c=value[valid],cmap=cmap,vmin=0,vmax=len(colors)-1,s=2,lw=0,rasterized=True)
        else:
            im=ax.scatter(*self.xy[valid].T,c=value[valid],cmap='viridis',vmin=0,vmax=vmax,s=2,lw=0,rasterized=True)
            ax.figure.colorbar(im,ax=ax,shrink=.55,pad=.015)
        ax.set_aspect('equal');ax.set_title(title,fontsize=9);ax.set_xticks([]);ax.set_yticks([])
        for spine in ax.spines.values():spine.set_visible(False)
        x=self.xy[self.show,0];y=self.xy[self.show,1];sx=x.min()+.07*np.ptp(x);sy=y.min()+.07*np.ptp(y)
        length=min(100.,.18*np.ptp(x));ax.plot([sx,sx+length],[sy,sy],color='#333333',lw=1.2);ax.text(sx+length/2,sy+.025*np.ptp(y),f'{length:g} µm',ha='center',fontsize=6)
        if categorical and names is not None:
            from matplotlib.lines import Line2D
            handles=[Line2D([],[],color=colors[j%len(colors)],marker='o',ls='',ms=3,label=n) for j,n in enumerate(names)]
            ax.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,-.01),ncol=2,fontsize=6)

    def spatial_comparison(self):
        fig,axes=plt.subplots(1,4,figsize=(12,3.3),layout='constrained')
        self.panel(axes[0],pd.Categorical(self.adata.obs.cell_type,categories=TYPES).codes,'Cell types',True,names=TYPES)
        self.panel(axes[1],self.truth,'Ground truth',True,names=['Tumor core','Normal exterior','Edge A','Edge B'])
        self.panel(axes[2],self.old_labels,'CCC-only GNN · raw K=4',True,names=[f'Niche {i+1}' for i in range(4)])
        self.panel(axes[3],self.labels,f'{self.model_label} · raw K={self.summary["selected_K"]}',True,names=[f'Niche {i+1}' for i in range(self.summary['selected_K'])])
        return self.save(fig,'spatial_ground_truth_and_predictions')

    def communication_recovery(self):
        fig,axes=plt.subplots(2,4,figsize=(12,6),layout='constrained')
        for j in range(4):
            spec=self.specs.iloc[j];title=f'MI-{j+1} · {spec.sender} → {spec.receiver}'
            self.panel(axes[0,j],self.true_signal[:,j],title+'\nObserved CCC',vmax=self.limits[j])
            self.panel(axes[1,j],self.recovered_signal[:,j],'Recovered ZH · all 32 programs',vmax=self.limits[j])
        return self.save(fig,'four_MI_communication_recovery')

    def expression_recovery(self):
        # Full-tissue descriptive signatures and separate held-out errors.
        # The purged test fold contains no core centers, so it cannot establish
        # generalization of core signatures. Never fill that gap with fake data.
        design=self.design[self.design.MI>0];names=design.gene.to_numpy();observed=[];predicted=[]
        for niche in np.unique(self.labels):
            a=[];b=[]
            for row in design.itertuples():
                t=TYPES.index(row.receiver);g=self.adata.var_names.get_loc(row.gene)
                r=np.flatnonzero((self.labels==niche)&self.present[:,t])
                reference=float(self.target[self.present[:,t],t,g].mean())
                a.append(float(self.target[r,t,g].mean()-reference) if len(r) else np.nan)
                b.append(float(self.prediction[r,t,g].mean()-reference) if len(r) else np.nan)
            observed.append(a);predicted.append(b)
        observed=np.asarray(observed);predicted=np.asarray(predicted);limit=max(float(np.nanmax(np.abs(np.r_[observed,predicted]))),1e-6)
        np.savez(self.output/'niche_gene_signatures.npz',observed=observed,predicted=predicted,genes=names,niches=np.unique(self.labels)+1)
        fig=plt.figure(figsize=(12,7),layout='constrained');grid=fig.add_gridspec(3,2,height_ratios=[1,1,1.1]);axes=[fig.add_subplot(grid[0,:]),fig.add_subplot(grid[1,:])]
        for ax,value,title in zip(axes,(observed,predicted),('All-center niche gene signature · observed','All-center niche gene signature · reconstructed')):
            im=ax.imshow(value,aspect='auto',cmap='RdBu_r',vmin=-limit,vmax=limit)
            ax.set_yticks(range(len(value)),[f'Niche {i+1}' for i in np.unique(self.labels)])
            ax.set_xticks(range(len(names)),[f'MI{r.MI} {"↑" if r.direction>0 else "↓"}{i%6+1}' for i,r in enumerate(design.itertuples())],rotation=60,ha='right',fontsize=6)
            ax.set_title(title,fontsize=9)
            for j in (5.5,11.5,17.5):ax.axvline(j,c='#333333',lw=.6)
        fig.colorbar(im,ax=axes,label='Weighted log-expression · centered by observed gene mean',shrink=.7)
        ax=fig.add_subplot(grid[2,0]);old=pd.read_csv(Path(self.summary['frozen_control'])/'expression_gene_metrics.csv')
        def response(frame):return frame[(frame.kind=='synthetic_response')&(frame.cell_type==frame.designed_receiver)].groupby('MI')
        baseline=response(self.gene_scores).baseline_RMSE.mean();frozen=response(old).RMSE.mean();joint=response(self.gene_scores).RMSE.mean();x=np.arange(4)
        selected_label='Joint decoder' if self.summary['joint_model_accepted'] else 'Retained frozen-Z decoder'
        for shift,values,color,label in [(-.24,baseline,'#a7b3ba','Type + library baseline'),(0,frozen,COLORS[0],'Frozen-Z decoder'),(.24,joint,COLORS[2],selected_label)]:ax.bar(x+shift,values.reindex(range(1,5)),.23,color=color,label=label)
        ax.set(xticks=x,xticklabels=[f'MI-{j+1}' for j in range(4)],ylabel='Mean gene RMSE',title='Held-out receiver responses · six genes per MI');ax.legend(fontsize=6)
        ax=fig.add_subplot(grid[2,1]);counts=np.bincount(self.labels[self.test],minlength=len(observed));ax.bar(np.arange(len(counts)),counts,color=COLORS[0]);ax.set(xticks=np.arange(len(counts)),xticklabels=[f'Niche {i+1}' for i in range(len(counts))],ylabel='Expression-test centers',title='Held-out coverage · no core-center evaluation')
        pd.DataFrame({'niche':np.arange(len(counts))+1,'expression_test_centers':counts}).to_csv(self.output/'expression_test_niche_coverage.csv',index=False)
        return self.save(fig,'observed_and_reconstructed_niche_genes')

    def niche_cell_types(self):
        fig,ax=plt.subplots(figsize=(7,3.5),layout='constrained');self._composition(ax)
        return self.save(fig,'niche_cell_type_composition')

    def _composition(self,ax):
        values=self.composition.to_numpy();im=ax.imshow(values,aspect='auto',cmap='Blues',vmin=0,vmax=1)
        ax.set_xticks(range(len(TYPES)),TYPES,rotation=25,ha='right',fontsize=7);ax.set_yticks(range(len(values)),[f'Niche {i}' for i in self.composition.index]);ax.set_title('Spatially weighted cell-type composition')
        for i in range(len(values)):
            for j in range(len(TYPES)):ax.text(j,i,f'{values[i,j]:.0%}',ha='center',va='center',fontsize=7,color='white' if values[i,j]>.6 else '#333333')
        ax.figure.colorbar(im,ax=ax,shrink=.65,label='Fraction')

    def curves(self):
        pre=pd.read_csv(self.source/'gnn/training.csv');joint=pd.read_csv(self.output/'joint_training.csv')
        fig,axes=plt.subplots(1,3,figsize=(11,3),layout='constrained')
        for name,color in [('train',COLORS[0]),('validation',COLORS[1])]:
            axes[0].plot(pre.epoch,pre[name+'_ccc'],color=color,label=name)
            axes[1].plot(joint.epoch,joint[name+'_ccc'],color=color,label=name)
            axes[2].plot(joint.epoch,joint[name+'_gene'],color=color,label=name)
        axes[0].set(title='CCC pretraining · NMF initialization',xlabel='Epoch',ylabel='Scaled CCC Huber (δ=3)')
        axes[1].set(title='Joint training · CCC task',xlabel='Joint epoch',ylabel='Scaled CCC Huber (δ=3)')
        axes[1].axhline(self.summary['validation_CCC_before']*(1+self.summary['CCC_validation_tolerance']),c='#999999',ls=':',label='Validation gate')
        axes[2].set(title='Joint training · expression task',xlabel='Joint epoch',ylabel='Scaled expression Huber (δ=1)')
        for ax in axes:ax.legend(fontsize=6)
        for ax in axes[1:]:ax.axvline(self.summary['joint_best_epoch'],c='#777777',ls='--',lw=.7)
        status='Joint checkpoint accepted' if self.summary['joint_model_accepted'] else 'No eligible improvement: baseline retained'
        fig.supxlabel(f'{status} · selected epoch {self.summary["joint_best_epoch"]} · relative Z change {self.summary["Z_relative_change"]:.2%}',fontsize=8)
        return self.save(fig,'CCC_and_gene_training_losses')

    def summary_figure(self):
        fig,axes=plt.subplots(3,4,figsize=(12,9.5),layout='constrained')
        self.panel(axes[0,0],pd.Categorical(self.adata.obs.cell_type,categories=TYPES).codes,'a  Simulated cell types',True,names=TYPES)
        self.panel(axes[0,1],self.truth,'b  Ground-truth niches',True,names=['Tumor core','Normal exterior','Edge A','Edge B'])
        from .evaluation import label_mapping,mapped
        train=np.load(self.output/'expression_split_rows.npz')['train']
        old_map=label_mapping(self.old_labels,self.truth,train);new_map=label_mapping(self.labels,self.truth,train)
        (self.output/'summary_plot_color_mapping.json').write_text(json.dumps({'fit_rows':'expression training blocks only; display/evaluation only','CCC_only':old_map,'joint':new_map},indent=2))
        self.panel(axes[0,2],mapped(self.old_labels,old_map),'c  CCC-only GNN · K=4',True)
        self.panel(axes[0,3],mapped(self.labels,new_map),f'd  {self.model_label} · K={self.summary["selected_K"]}',True)
        for j in range(4):
            self.panel(axes[1,j],self.mi[:,j],f'{chr(101+j)}  True MI-{j+1} activity',vmax=max(float(np.quantile(self.mi[:,j],.99)),1e-8))
            self.panel(axes[2,j],self.recovered_signal[:,j],f'{chr(105+j)}  Recovered MI-{j+1} · ZH',vmax=self.limits[j])
        mode='joint' if self.summary['joint_model_accepted'] else 'frozen-Z'
        fig.suptitle(f'Tumor-edge communication: simulated truth and CCC-priority {mode} reconstruction',fontsize=12)
        # Actual held-out scores, not idealized success labels.
        frame=pd.read_csv(self.output/'metrics.csv');rows=frame[frame.evaluation=='heldout_test']
        scores=' | '.join(f'{r.method}: ARI {r.overall_ARI:.3f}, edge F1 {r.edge_F1:.3f}' for r in rows.itertuples())
        fig.supxlabel(scores+'\nCluster colors matched on training centers for display only; raw IDs remain in the first figure and tables. True activity and projected CCC have distinct units.',fontsize=7)
        return self.save(fig,'simulation_and_model_summary')
