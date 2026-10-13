"""Publication grid: actual raw cluster outputs, never fabricated/aligned success maps."""
from pathlib import Path
import json
import numpy as np,pandas as pd
import matplotlib.pyplot as plt
from ..gnn.reporting import style,COLORS
NICHES=['Tumor core','Normal exterior','Edge A (MI-1)','Edge B (MI-2)']

class SimulationReport:
    def __init__(self,output,load_results=True):
        import anndata as ad
        self.output=Path(output);self.adata=ad.read_h5ad(self.output/'injected.h5ad')
        self.cells=self.adata.obs;self.xy=np.asarray(self.adata.obsm['spatial']);self.truth=self.cells.true_niche.to_numpy()
        if load_results:
            self.labels=pd.read_csv(self.output/'raw_clusters.csv');self.metrics=pd.read_csv(self.output/'metrics.csv')
            self.Z=np.load(self.output/'gnn/Z.npy',mmap_mode='r');self.H=np.load(self.output/'gnn/H.npy');self.B=np.load(self.output/'gnn/B.npy')
            self.match=pd.read_csv(self.output/'MI_matches.csv')
        self.true_mi=np.load(self.output/'ground_truth/MI_activity.npy')
        specs_path=self.output/'ground_truth/MI_programs.csv'
        self.mi_specs=pd.read_csv(specs_path) if specs_path.exists() else pd.DataFrame({'MI':range(1,self.true_mi.shape[1]+1),'name':['Edge A','Edge B'][:self.true_mi.shape[1]]})
        self.features=pd.read_csv(self.output/'input/features.csv');self.cfg=json.loads((self.output/'config.json').read_text())
        self.figures=self.output/'figures';self.figures.mkdir(exist_ok=True);style()
        folds=np.load(self.output/'input/folds.npy');test=np.flatnonzero(folds==2)
        # One complete held-out replicate, or the complete single tissue with
        # all-cell transductive plots explicitly separated from held-out metrics.
        sample=self.cells.sample_id.iloc[test[0]];self.show=(self.cells.sample_id.to_numpy()==sample)
    def save(self,fig,name):
        fig.savefig(self.figures/(name+'.png'),dpi=self.cfg['figure_dpi'],bbox_inches='tight');fig.savefig(self.figures/(name+'.pdf'),bbox_inches='tight')
        return fig
    def spatial(self,ax,values,title,categorical=True):
        xy=self.xy[self.show];v=np.asarray(values)[self.show]
        if categorical:ax.scatter(*xy.T,c=np.asarray(COLORS)[v.astype(int)],s=2,lw=0,rasterized=True)
        else:
            p=ax.scatter(*xy.T,c=v,s=2,lw=0,rasterized=True,cmap='magma',vmin=0,vmax=max(float(np.quantile(v,.99)),1e-8));plt.colorbar(p,ax=ax,shrink=.7)
        ax.set(title=title,xlabel='x (µm)',ylabel='y (µm)');ax.set_aspect('equal')
        xmin,ymin=xy.min(0);span=np.ptp(xy[:,0]);bar=100 if span>300 else 20
        ax.plot([xmin,xmin+bar],[ymin,ymin],c='#333333',lw=1.5);ax.text(xmin+bar/2,ymin+.025*np.ptp(xy[:,1]),f'{bar} µm',ha='center',fontsize=6)
    def ground_truth(self):
        from .generation import TYPES
        count=self.true_mi.shape[1];fig,axes=plt.subplots(2,3,figsize=(11,7),layout='constrained');axes=axes.ravel()
        self.spatial(axes[0],pd.Categorical(self.cells.cell_type,categories=TYPES).codes,'Cell types')
        self.spatial(axes[1],self.truth,'Ground-truth niches')
        for j in range(count):self.spatial(axes[j+2],self.true_mi[:,j],f'True MI-{j+1} · {self.mi_specs.name.iloc[j]}',False)
        for ax in axes[count+2:]:ax.set_visible(False)
        axes[0].legend(handles=[plt.Line2D([],[],color=COLORS[i],marker='o',ls='',ms=4,label=t) for i,t in enumerate(TYPES)],loc='upper center',bbox_to_anchor=(.5,-.15),fontsize=6,ncol=2)
        axes[1].legend(handles=[plt.Line2D([],[],color=COLORS[i],marker='o',ls='',ms=4,label=t) for i,t in enumerate(NICHES)],loc='upper center',bbox_to_anchor=(.5,-.15),fontsize=6,ncol=2)
        return self.save(fig,'ground_truth')
    def main_figure(self):
        names=['Cell Composition','Expression PCA','CCC-AE','CCC_Phe GNN (fixed K)']
        fig,axes=plt.subplots(1,5,figsize=(16,3.8),layout='constrained')
        self.spatial(axes[0],self.truth,'a  Ground Truth')
        for i,name in enumerate(names):self.spatial(axes[i+1],self.labels[name].to_numpy(),f'{chr(98+i)}  '+['Cell Composition','Expression Clustering','CCC-AE','CCC_Phe GNN'][i])
        for ax,names_here in zip(axes,[NICHES]+[[f'Raw cluster {i}' for i in range(self.cfg['fixed_k'])]]*4):
            ax.legend(handles=[plt.Line2D([],[],color=COLORS[i],marker='o',ls='',ms=3,label=t) for i,t in enumerate(names_here)],loc='upper center',bbox_to_anchor=(.5,-.18),fontsize=6,ncol=2)
        fig.suptitle('Fixed-K comparison: unaligned raw cluster IDs; colors across methods do not imply matching',fontsize=10)
        return self.save(fig,'main_comparison')
    def recovered_programs(self):
        count=self.true_mi.shape[1];fig,axes=plt.subplots(count,2,figsize=(8,3.25*count),layout='constrained',squeeze=False)
        for i in range(count):
            row=self.match[(self.match.MI==i+1)&(self.match.scope=='test_all')].iloc[0];p=int(row.program)-1
            self.spatial(axes[i,0],self.true_mi[:,i],f'True MI-{i+1} · {self.mi_specs.name.iloc[i]}',False)
            self.spatial(axes[i,1],self.Z[:,p],f'Learned P{p+1:02d}; dictionary cosine={row.dictionary_cosine:.2f}',False)
        return self.save(fig,'true_vs_recovered_programs')
    def reconstruction(self):
        history=pd.read_csv(self.output/'gnn/training.csv');keys=['ccc']+(['composition'] if 'train_composition' in history else [])
        fig,axes=plt.subplots(1,len(keys),figsize=(5*len(keys),3),layout='constrained',squeeze=False)
        for ax,key in zip(axes.ravel(),keys):
            for split,c in [('train',COLORS[0]),('validation',COLORS[1])]:ax.plot(history.epoch,history[split+'_'+key],label=split,c=c)
            ax.set(xlabel='Epoch',ylabel=('Scaled '+self.cfg.get('ccc_error','huber')) if key=='ccc' else 'KL divergence',title=key+' reconstruction');ax.legend()
        return self.save(fig,'training_curves')
    def dictionary(self):
        import seaborn as sns
        match=self.match[self.match.scope=='test_all'];programs=match.program.to_numpy(int)-1
        true_h=np.load(self.output/'ground_truth/MI_dictionary.npy');chosen=np.unique(np.concatenate([np.flatnonzero(true_h.sum(0)),*[np.argsort(-self.H[p])[:6] for p in programs]]))
        f=self.features.iloc[chosen];labels=(f.sender+'→'+f.receiver+' | '+f.ligand+'–'+f.receptor).tolist()
        fig,ax=plt.subplots(figsize=(max(9,len(chosen)*.23),3.8),layout='constrained')
        sns.heatmap(np.vstack((true_h[:,chosen],self.H[programs][:,chosen])),cmap='Blues',ax=ax,xticklabels=labels,yticklabels=[f'True MI-{i+1} · {self.mi_specs.name.iloc[i]}' for i in range(len(true_h))]+[f'Learned P{i+1:02d}' for i in programs],cbar_kws={'label':'Native dictionary loading'})
        ax.tick_params(axis='x',rotation=90,labelsize=6);return self.save(fig,'MI_dictionary_recovery')
    def composition_audit(self):
        import seaborn as sns
        frame=pd.read_csv(self.output/'radial_composition_audit.csv');auxiliary=self.cfg['lambda_comp']>0
        fig,axs=plt.subplots(1,2 if auxiliary else 1,figsize=(10 if auxiliary else 6,3.8),layout='constrained',squeeze=False);axes=axs.ravel()
        for name,part in frame.groupby('cell_type'):
            axes[0].plot(part.radial_bin,part.A,label=name);axes[0].plot(part.radial_bin,part.B,ls='--',lw=1)
        axes[0].set(xlabel='Radial bin',ylabel='Pair-weighted composition',title='A solid / B dashed');axes[0].legend(fontsize=6)
        from .generation import TYPES
        if auxiliary:
            sns.heatmap(self.B,cmap='Blues',ax=axes[1],xticklabels=TYPES,yticklabels=False,cbar_kws={'label':'Composition fraction'})
            axes[1].set(title='Learned B (32 × 5)',ylabel='Communication programs');axes[1].tick_params(axis='x',rotation=65)
        return self.save(fig,'composition_control')
    def adaptive_niches(self):
        fig,ax=plt.subplots(figsize=(5,4),layout='constrained');self.spatial(ax,self.labels['CCC_Phe GNN (selected K)'].to_numpy(),'GNN unsupervised K selection')
        return self.save(fig,'adaptive_niches')

    def summary_figure(self):
        from .summary_figure import summary_figure
        return summary_figure(self)
