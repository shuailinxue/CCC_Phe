"""Compact Python report: tissue recovery, group specificity, programs, training.

Figure contract: test whether composition-matched communication-state changes
are recovered in unseen slices. PNG/PDF/SVG source data stay on NAS. No outcome
is assumed; raw labels and failed recovery are retained in the result tables.
"""
from pathlib import Path
import json
import anndata as ad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter
from .paired_tissues import TYPES,NICHES

# Distinct warm/cool accents avoid using red versus green for the disease pair.
COLORS=['#3976AC','#66A49A','#D4A72C','#9B85B8','#D55E55','#7046A0','#8B8B8B','#BBBBBB']
CELL_COLORS=['#3976AC','#D39B37','#A482B8','#62A69E','#D36770']
ACTIVITY_CMAP=LinearSegmentedColormap.from_list('communication_activity',
    ['#E0E4E8','#C7B6D7','#A17AB8','#74418F','#3C1459'])


class PairedReport:
    def __init__(self,output):
        self.output=Path(output);self.figures=self.output/'figures';self.figures.mkdir(exist_ok=True)
        self.a=ad.read_h5ad(self.output/'tissues.h5ad');self.cells=pd.read_csv(self.output/'assignments.csv.gz')
        if not np.array_equal(self.a.obs.cell_id.to_numpy(),self.cells.cell_id.to_numpy()):raise ValueError('Report cell order mismatch')
        self.summary=json.loads((self.output/'summary.json').read_text())
        self.enrichment=pd.read_csv(self.output/'disease_enrichment.csv')
        self.metrics=pd.read_csv(self.output/'metrics.csv')
        self.top=pd.read_csv(self.output/'niche_top10_CCC.csv')
        self.recovery=pd.read_csv(self.output/'program_recovery.csv')
        plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10.5,'axes.titlesize':12,
                             'axes.labelsize':10.5,'xtick.labelsize':10,'ytick.labelsize':10,
                             'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.8,
                             'figure.dpi':120,'savefig.facecolor':'white',
                             'pdf.fonttype':42,'svg.fonttype':'none','legend.frameon':False})

    def save(self,fig,name):
        for ext in ('png','pdf','svg'):fig.savefig(self.figures/(name+'.'+ext),dpi=450,bbox_inches='tight',pad_inches=.12)
        return fig

    def scatter(self,ax,rows,values,title,continuous=False,vmax=None,palette=None):
        xy=self.a.obsm['spatial'][rows]
        if continuous:
            im=ax.scatter(*xy.T,c=values,s=9,cmap=ACTIVITY_CMAP,vmin=0,vmax=vmax,rasterized=True,lw=0)
        else:
            colors=palette or COLORS
            point_colors=np.asarray([colors[int(v)%len(colors)] if v>=0 else '#888888' for v in values])
            im=ax.scatter(*xy.T,c=point_colors,s=11,lw=0,rasterized=True)
        ax.set_aspect('equal');ax.set_title(title,pad=10);ax.set_xticks([]);ax.set_yticks([])
        for s in ax.spines.values():s.set_visible(False)
        radius=float(np.abs(self.a.obsm['spatial']).max())
        ax.set(xlim=(-1.04*radius,1.04*radius),ylim=(-1.15*radius,1.04*radius))
        x,y=-.98*radius,-1.03*radius
        ax.plot([x,x+100],[y,y],color='#333333',lw=1.5)
        ax.text(x+50,y-.025*radius,'100 µm',ha='center',va='top',fontsize=9)
        return im

    @staticmethod
    def panel_label(ax,letter):
        ax.text(-.025,1.035,letter,transform=ax.transAxes,weight='bold',fontsize=13,va='bottom')

    def cell_types(self):
        fig,axes=plt.subplots(1,2,figsize=(8.6,4.5))
        fig.subplots_adjust(left=.035,right=.98,bottom=.18,top=.90,wspace=.10)
        types=pd.Categorical(self.cells.cell_type,categories=TYPES).codes
        for col,condition in enumerate(('Healthy','Cancer')):
            rows=np.flatnonzero((self.cells.fold==2)&(self.cells.condition==condition))
            self.scatter(axes[col],rows,types[rows],condition,palette=CELL_COLORS)
            self.panel_label(axes[col],chr(97+col))
        fig.legend(handles=[Line2D([],[],marker='o',ls='',color=CELL_COLORS[i],label=t,markersize=5)
                            for i,t in enumerate(TYPES)],loc='lower center',ncol=3,
                   bbox_to_anchor=(.5,.015),fontsize=10,columnspacing=1.5,handletextpad=.45)
        return self.save(fig,'heldout_cell_types')

    def spatial(self):
        fig,axes=plt.subplots(2,2,figsize=(8.6,8.0))
        fig.subplots_adjust(left=.06,right=.99,bottom=.16,top=.94,wspace=.08,hspace=.18)
        for col,condition in enumerate(('Healthy','Cancer')):
            rows=np.flatnonzero((self.cells.fold==2)&(self.cells.condition==condition))
            self.scatter(axes[0,col],rows,self.cells.true_niche.to_numpy()[rows],condition)
            self.scatter(axes[1,col],rows,self.cells.evaluation_aligned_niche.to_numpy()[rows],'')
        for ax,letter in zip(axes.ravel(),'abcd'):self.panel_label(ax,letter)
        fig.text(.02,.735,'Ground truth',rotation=90,va='center',ha='center',weight='bold',fontsize=11)
        fig.text(.02,.345,f'GNN prediction · K={self.summary["selected_K"]}',rotation=90,va='center',ha='center',weight='bold',fontsize=11)
        handles=[Line2D([],[],marker='o',ls='',color=COLORS[i],label=t,markersize=5) for i,t in enumerate(NICHES)]
        if (self.cells.evaluation_aligned_niche<0).any():
            handles.append(Line2D([],[],marker='o',ls='',color='#888888',label='Unmatched cluster',markersize=5))
        fig.legend(handles=handles,loc='lower center',ncol=2,bbox_to_anchor=(.54,.015),
                   fontsize=10,columnspacing=1.8,handletextpad=.45)
        return self.save(fig,'heldout_tissue_comparison')

    def groups(self):
        f=pd.read_csv(self.output/'sample_niche_fractions.csv').sort_values(['fold','sample_id'])
        composition=pd.read_csv(self.output/'niche_composition.csv',index_col=0)
        columns=[str(i) for i in range(1,self.summary['selected_K']+1)]
        fig,axes=plt.subplots(1,2,figsize=(9.5,4.4),layout='constrained',gridspec_kw={'width_ratios':[1.2,1]})
        candidates=set(self.enrichment.loc[self.enrichment.disease_specific_candidate,'niche'])
        for i,col in enumerate(columns):
            if int(col) in candidates:axes[0].axvspan(i-.42,i+.42,color='#F7E8E8',zorder=0,lw=0)
            for pair in sorted(self.cells.pair_id.unique()):
                pair_f=f[f.sample_id.str.startswith(f'pair{pair+1}_')]
                healthy=pair_f[pair_f.condition=='Healthy'].iloc[0];cancer=pair_f[pair_f.condition=='Cancer'].iloc[0]
                offset=(pair-float(self.cells.pair_id.max())/2)*.025
                axes[0].plot([i-.13+offset,i+.13+offset],[healthy[col],cancer[col]],color='#B6BCC2',lw=.8,zorder=1)
                for value,x,color in [(healthy,i-.13+offset,'#287E8D'),(cancer,i+.13+offset,'#D15B64')]:
                    test=int(value['fold'])==2
                    axes[0].scatter(x,value[col],s=35 if test else 22,marker='D' if test else 'o',
                                    c=color,edgecolor='#333333' if test else 'white',lw=.6,zorder=3)
        axes[0].set(xticks=np.arange(len(columns)),xticklabels=['N'+v for v in columns],
                    xlabel='Learned niche',ylabel='Cells per slice (%)',title='Niche abundance',
                    ylim=(-.025,max(.4,float(f[columns].to_numpy().max())*1.22)))
        axes[0].yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
        axes[0].legend(handles=[Line2D([],[],marker='o',ls='',color=c,label=l,markersize=5)
                                for c,l in [('#287E8D','Healthy'),('#D15B64','Cancer')]],loc='upper right',fontsize=9)
        im=axes[1].imshow(composition,aspect='auto',cmap='Blues',vmin=0,vmax=1)
        axes[1].set(xticks=np.arange(5),xticklabels=TYPES,yticks=np.arange(len(composition)),
                    yticklabels=['N'+str(i) for i in composition.index],ylabel='Learned niche',title='Cell-type composition (%)')
        for i in range(len(composition)):
            for j in range(5):
                v=composition.iloc[i,j]
                axes[1].text(j,i,f'{v*100:.0f}',ha='center',va='center',fontsize=10,color='white' if v>.5 else '#222222')
        plt.setp(axes[1].get_xticklabels(),rotation=40,ha='right')
        bar=fig.colorbar(im,ax=axes[1],shrink=.7,pad=.025);bar.ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
        for ax,letter in zip(axes,'ab'):self.panel_label(ax,letter)
        return self.save(fig,'group_specificity_and_composition')

    def programs(self):
        rows=np.flatnonzero((self.cells.fold==2)&(self.cells.condition=='Cancer'))
        true=np.load(self.output/'ground_truth/microenvironment_activity.npy')[rows]
        z=np.load(self.output/'gnn/Z.npy')[rows]
        fig,axes=plt.subplots(2,3,figsize=(9.5,6.5))
        fig.subplots_adjust(left=.055,right=.91,bottom=.04,top=.93,wspace=.10,hspace=.16)
        matched=[int(self.recovery.iloc[p].learned_program)-1 for p in (5,6,7)]
        limits=[max(float(np.quantile(true[:,5:8],.99)),1e-8),max(float(np.quantile(z[:,matched],.99)),1e-8)]
        row_images={}
        for col,p in enumerate((5,6,7)):
            match=self.recovery.iloc[p];j=int(match.learned_program)-1
            for row,value,title in [(0,true[:,p],['Interface A','Interface B','Immune response'][col]),
                                     (1,z[:,j],f'Z{j+1} · ρ={match.heldout_activity_spearman:.2f}')]:
                row_images[row]=self.scatter(axes[row,col],rows,value,title,True,vmax=limits[row])
        for ax,letter in zip(axes.ravel(),'abcdef'):self.panel_label(ax,letter)
        fig.text(.018,.72,'Ground truth',rotation=90,va='center',ha='center',weight='bold',fontsize=11)
        fig.text(.018,.26,'Learned activity',rotation=90,va='center',ha='center',weight='bold',fontsize=11)
        for row,y in [(0,.60),(1,.14)]:
            cax=fig.add_axes([.935,y,.014,.24]);fig.colorbar(row_images[row],cax=cax,label='Activity')
        return self.save(fig,'true_and_recovered_disease_programs')

    def training(self):
        history=pd.read_csv(self.output/'gnn/training.csv');selection=pd.read_csv(self.output/'gnn/K_selection.csv')
        fig,axes=plt.subplots(1,3,figsize=(10,3.4),layout='constrained')
        axes[0].plot(history.epoch,history.train_ccc,label='Training',color=COLORS[0])
        axes[0].plot(history.epoch,history.validation_ccc,label='Validation slices',color=COLORS[4])
        axes[0].set(xlabel='Epoch',ylabel='Scaled Huber loss',title='CCC reconstruction');axes[0].legend(fontsize=9)
        heldout=json.loads((self.output/'gnn/heldout_metrics.json').read_text())
        axes[0].axvline(heldout['best_epoch'],ls=':',color='#777777',lw=1)
        axes[1].plot(selection.K,selection.selection_score,'o-',color=COLORS[0])
        axes[1].axvline(self.summary['selected_K'],ls='--',color=COLORS[4]);axes[1].set(xlabel='K',ylabel='Validation selection score',title='Unsupervised niche selection')
        target=np.load(self.output/'input/ccc.npy',mmap_mode='r');pred=np.load(self.output/'reconstruction.npy',mmap_mode='r')
        mask=np.load(self.output/'input/preprocessing.npz')['feature_mask'];rows=np.flatnonzero(self.cells.fold==2)
        rng=np.random.default_rng(234);r=rng.choice(rows,40000);c=rng.choice(np.flatnonzero(mask),40000)
        x=np.log1p(target[r,c]);y=np.log1p(pred[r,c]);im=axes[2].hexbin(x,y,gridsize=40,mincnt=1,cmap='Blues',bins='log')
        limit=max(x.max(),y.max());axes[2].plot([0,limit],[0,limit],color='#777777',lw=.7)
        axes[2].set(xlabel='Observed log(1 + CCC)',ylabel='Predicted log(1 + CCC)',title='Held-out reconstruction');fig.colorbar(im,ax=axes[2],label='Entries',shrink=.8)
        for ax,letter in zip(axes,'abc'):self.panel_label(ax,letter)
        return self.save(fig,'training_K_and_reconstruction')

    def candidate_communications(self):
        ids=self.enrichment.loc[self.enrichment.disease_specific_candidate,'niche']
        return self.top[self.top.niche.isin(ids)].reset_index(drop=True)
